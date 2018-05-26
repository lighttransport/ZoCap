#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

static void RotatePoint(const cv::Point2f& src, cv::Point2f& dst,
                        const float radian, const cv::Point2f& center) {
    const float s = std::sin(radian);
    const float c = std::cos(radian);
    dst.x = c * src.x + s * src.y + center.x;
    dst.y = -s * src.x + c * src.y + center.y;
}

static int CalcID(const cv::Point2f& center,
                  const std::vector<cv::Point2f>& outer_pts,
                  const std::vector<cv::Point2f>& inner_pts,
                  const float radius) {
    float left_pt_x = std::numeric_limits<float>::max();
    size_t left_pt_idx = 0;
    for (size_t i = 0; i < outer_pts.size(); i++) {
        if (outer_pts[i].x < left_pt_x) {
            left_pt_x = outer_pts[i].x;
            left_pt_idx = i;
        }
    }

    // Base point
    const cv::Point2f base_pt = center - outer_pts[left_pt_idx];

    int bit_outer = 0, bit_inner = 0;
    const float pi = static_cast<float>(CV_PI);
    for (int angle_idx = 0; angle_idx < 6; angle_idx++) {
        bit_outer <<= 1;
        bit_inner <<= 1;

        {
            // Outer points
            const float rad = static_cast<float>(angle_idx) * pi / 3.f;
            cv::Point2f rotated_pt;
            RotatePoint(base_pt, rotated_pt, rad, center);
            for (size_t i = 0; i < outer_pts.size(); i++) {
                const float dd = cv::norm(outer_pts[i] - rotated_pt);
                if (dd < radius / 4) {
                    bit_outer += 1;
                    continue;
                }
            }
        }

        {
            // Outer points
            const float rad = (static_cast<float>(angle_idx) + 0.5f) * pi / 3.f;
            cv::Point2f rotated_pt;
            cv::Point2f inner_base_pt = base_pt * (1.f / std::sqrt(3.f));
            RotatePoint(inner_base_pt, rotated_pt, rad, center);
            for (size_t i = 0; i < inner_pts.size(); i++) {
                const float dd = cv::norm(inner_pts[i] - rotated_pt);
                if (dd < radius / 4) {
                    bit_inner += 1;
                    continue;
                }
            }
        }
    }

    // Normalize id (Maximum value with rotation)
    int max_id = 0;
    for (int angle_idx = 0; angle_idx < 6; angle_idx++) {
        // Update maximum id
        int id = (bit_outer << 6) | bit_inner;
        if (max_id < id) {
            max_id = id;
        }
        // Bit shift
        bit_outer <<= 1;
        if (bit_outer & 0x40) {
            bit_outer |= 1;
        }
        bit_outer &= 0x3f;

        bit_inner <<= 1;
        if (bit_inner & 0x40) {
            bit_inner |= 1;
        }
        bit_inner &= 0x3f;
    }

    return max_id;
}

static int DescriptZozoId(const std::vector<cv::Point2f> &points,
                          const cv::Point2f& semi_center,
                          cv::Point2f& center_pt) {
    // Search the most inside point
    float min_dist = std::numeric_limits<float>::max();
    size_t min_dist_idx = 0;
    for (size_t i = 0; i < points.size(); i++) {
        const float dist = cv::norm(points[i] - semi_center);
        if (dist < min_dist) {
            min_dist = dist;
            min_dist_idx = i;
        }
    }
    center_pt = points[min_dist_idx];

    // Distances from the most inside point
    std::vector<float> point_dists;
    for (size_t i = 0; i < points.size(); i++) {
        const float dist = cv::norm(points[i] - center_pt);
        point_dists.push_back(dist);
    }

    // Search the most outside point
    float max_dist = 0.f;
    for (size_t i = 0; i < points.size(); i++) {
        max_dist = std::max(max_dist, point_dists[i]);
    }

    // Device into outer or inner
    std::vector<cv::Point2f> outer_pts, inner_pts;
    for (size_t i = 0; i < points.size(); i++) {
        if (i == min_dist_idx) {
            continue;
        }
        const float &dist = point_dists[i];
        if (1.f * dist / max_dist > 0.7f) {
            outer_pts.push_back(points[i]);
        } else {
            inner_pts.push_back(points[i]);
        }
    }
    if (outer_pts.size() < 2 || inner_pts.size() < 2) {
        return -1;
    }

    // Calculate ID
    int id = CalcID(center_pt, outer_pts, inner_pts, max_dist);
    return id;
}

inline void Apply2x3Mat(float& x, float& y, const cv::Mat& M) {
    float tmp_x = M.at<double>(0, 0) * x + M.at<double>(0, 1) * y +
                  M.at<double>(0, 2);
    float tmp_y = M.at<double>(1, 0) * x + M.at<double>(1, 1) * y +
                  M.at<double>(1, 2);
    x = tmp_x;
    y = tmp_y;
}

static void NormalizeEllipseWarping(std::vector<cv::Point2f>& points,
                                    const cv::RotatedRect& ellipse){
    cv::Mat M1 = getRotationMatrix2D(ellipse.center, -ellipse.angle, 1.0);
    cv::Mat M2 = getRotationMatrix2D(ellipse.center, ellipse.angle, 1.0);
    const float aspect = ellipse.size.width / ellipse.size.height;
    for (size_t i = 0; i < points.size(); i++) {
        float &x = points[i].x;
        float &y = points[i].y;
        Apply2x3Mat(x, y, M1);
        if (aspect < 1.f) {
            y = (y - ellipse.center.y) * aspect + ellipse.center.y;
        } else {
            x = (x - ellipse.center.x) / aspect + ellipse.center.x;
        }
        Apply2x3Mat(x, y, M2);
        points[i].x = x;
        points[i].y = y;
    }
}

static void DetectZozoMakrders(const cv::Mat& img,
                               std::vector<cv::Point2f>& out_points,
                               std::vector<int>& out_ids, bool show_debug) {
    out_points.clear();
    out_ids.clear();

    // Binarize for contour detection
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
//     cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.0);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // Detect contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(gray, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);

    cv::Mat debug_frame;
    if (show_debug) {
        debug_frame = img.clone();
    }
    for (size_t cont_idx = 0; cont_idx < contours.size(); cont_idx++) {
        const std::vector<cv::Point> cont = contours[cont_idx];

        // Ignore wrong contours
        {
            const cv::RotatedRect rect = cv::minAreaRect(cont);
            const float rect_size = rect.size.width * rect.size.height;
            const float rect_aspect = rect.size.width / rect.size.height;
            if (100000.f < rect_size) {
                continue;
            }
            if (rect_aspect < 0.4f or 1.6f < rect_aspect) {
                continue;
            }
        }

        // Approximate with Ellipse
        if (cont.size() <= 4) {
            continue;
        }
        const cv::RotatedRect ellipse = cv::fitEllipse(cont);
        int con = hierarchy[cont_idx][2];
        if (con == -1) {
            continue;
        }

        // Enumerate inner points
        std::vector<cv::Point2f> points;
        while (con != -1) {
            const cv::Rect rect = cv::boundingRect(contours[size_t(con)]);
            const cv::Point2f pt = (rect.br() + rect.tl()) * 0.5f;
            points.push_back(pt);
            con = hierarchy[size_t(con)][0];
        }

        // Normalize ellipse's warping
        NormalizeEllipseWarping(points, ellipse);

        cv::Point2f center_pt;
        int id = DescriptZozoId(points, ellipse.center, center_pt);
        if (id < 0) {
            continue;
        }

        // Register
        out_points.push_back(center_pt);
        out_ids.push_back(id);


        if (show_debug) {
            // Original marker
            cv::ellipse(debug_frame, ellipse, cv::Scalar(0, 255, 0), 2);
            int con = hierarchy[cont_idx][2];
            while (con != -1) {
                drawContours(debug_frame, contours, con, cv::Scalar(0, 0, 255),
                             -1);
                con = hierarchy[size_t(con)][0];
            }
            // Normalized marker
//             cv::circle(debug_frame, ellipse.center,
//                        std::max(ellipse.size.width, ellipse.size.height) / 2,
//                        cv::Scalar(0, 255, 0), -1);
//             for (int i = 0; i < points.size(); i++) {
//                 cv::circle(debug_frame, points[i], 3, cv::Scalar(255, 0, 0), -1);
//             }
        }
    }

    cv::imshow("debug", debug_frame);
}

int main(int argc, char* argv[]){
    (void)argc, (void)argv;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Failed to open\n");
        return 0;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            printf("Failed to capture\n");
            continue;
        }

        // Detect
        std::vector<cv::Point2f> points;
        std::vector<int> ids;
        DetectZozoMakrders(frame, points, ids, true);

        // Draw IDs
        for (size_t i = 0; i < points.size(); i++) {
            // Center point
            cv::circle(frame, points[i], 3, cv::Scalar(0, 255, 0), -1);
            // ID
            std::stringstream ss;
            ss << ids[i];
            cv::putText(frame, ss.str(), points[i], cv::FONT_HERSHEY_PLAIN,
                        1.5, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("frame", frame);
        const char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
    }
    return 0;
}
