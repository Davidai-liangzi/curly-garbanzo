#include<opencv2/opencv.hpp>
#include <vector>
#include<iostream>
#include<opencv2/calib3d.hpp>
#include<algorithm>

using namespace cv;
using namespace std;

int main()
{

    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Error: Could not open camera" << endl;
        return -1;
    }


    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);


    cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) <<
        1000, 0, 640,    // fx=1000, 光心cx=640
        0, 1000, 360,    // fy=1000, 光心cy=360
        0, 0, 1);

    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_32F);

    float a4_width = 0.210f;
    float a4_height = 0.297f;

    vector<Point3f> object_points = {
        Point3f(-a4_width / 2, -a4_height / 2, 0),  // 左下角
        Point3f(a4_width / 2, -a4_height / 2, 0),   // 右下角
        Point3f(a4_width / 2, a4_height / 2, 0),    // 右上角
        Point3f(-a4_width / 2, a4_height / 2, 0)    // 左上角
    };

    int canny_low = 50;
    int canny_high = 150;
    int blur_size = 5;

    namedWindow("AR Visualization", WINDOW_AUTOSIZE);

    Mat frame, gray, blurred, edges;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    while (true) {
        // 读取帧
        capture >> frame;
        if (frame.empty()) {
            cerr << "Error: Failed to capture image" << endl;
            break;
        }


        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, Size(blur_size, blur_size), 0);
        Canny(blurred, edges, canny_low, canny_high);
        morphologyEx(edges, edges, MORPH_CLOSE, kernel);
        findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat img = frame.clone();
        vector<vector<Point>> rectangle_contours;

        // 轮廓
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area < 1000) {
                continue;
            }

            vector<Point> approx;
            double epsilon = 0.02 * arcLength(contours[i], true);
            approxPolyDP(contours[i], approx, epsilon, true);

            if (approx.size() == 4 && isContourConvex(approx)) {
                rectangle_contours.push_back(approx);
            }
        }


        if (!rectangle_contours.empty()) {
            auto largest_rect = *max_element(rectangle_contours.begin(),
                rectangle_contours.end(),
                [](const vector<Point>& a, const vector<Point>& b) {
                    return contourArea(a) < contourArea(b);
                });

            drawContours(img, vector<vector<Point>>{largest_rect}, -1, Scalar(0, 255, 0), 3);

            vector<Point> sorted_points = largest_rect;
            sort(sorted_points.begin(), sorted_points.end(),
                [](const Point& a, const Point& b) {
                    return a.y < b.y;
                });

            vector<Point> bottom_points(sorted_points.begin() + 2, sorted_points.end());
            sort(bottom_points.begin(), bottom_points.end(),
                [](const Point& a, const Point& b) {
                    return a.x < b.x;
                });

            vector<Point> top_points(sorted_points.begin(), sorted_points.begin() + 2);
            sort(top_points.begin(), top_points.end(),
                [](const Point& a, const Point& b) {
                    return a.x < b.x;
                });

            vector<Point2f> image_points = {
                bottom_points[0],  // 左下
                bottom_points[1],  // 右下
                top_points[1],     // 右上
                top_points[0]      // 左上
            };

            //姿态估计
            try {
                Mat rvec, tvec;
                bool success = solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

                if (success) {
                    float axis_length_x = a4_width / 3;
                    float axis_length_y = a4_height / 3;
                    float axis_length_z = 0.05f;

                    vector<Point3f> axis_points = {
                        Point3f(0, 0, 0),
                        Point3f(axis_length_x, 0, 0),
                        Point3f(0, axis_length_y, 0),
                        Point3f(0, 0, axis_length_z)
                    };

                    vector<Point2f> projected_points;
                    projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);

                    Point origin = projected_points[0];
                    Point x_axis = projected_points[1];
                    Point y_axis = projected_points[2];
                    Point z_axis = projected_points[3];

                    // 绘制坐标轴
                    arrowedLine(img, origin, x_axis, Scalar(0, 0, 255), 3);
                    arrowedLine(img, origin, y_axis, Scalar(0, 255, 0), 3);
                    arrowedLine(img, origin, z_axis, Scalar(255, 0, 0), 3);

                    putText(img, "X", x_axis, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                    putText(img, "Y", y_axis, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                    putText(img, "Z", z_axis, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);

                    circle(img, origin, 8, Scalar(0, 255, 255), -1);
                }
            } catch (const exception& e) {
                cerr << "Error in PnP: " << e.what() << endl;
            }
        }


        imshow("result", img);


        char key = waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    }


    capture.release();
    destroyAllWindows();
    return 0;
}