#include <opencv2/opencv.hpp>

int main() {
    // 读取图像
    cv::Mat img_BGR = cv::imread("C:/Users/16968/Desktop/123/bed_pic.png");

    // 转换为HSV颜色空间
    cv::Mat img_HSV;
    cv::cvtColor(img_BGR, img_HSV, cv::COLOR_BGR2HSV);

    // 分割HSV通道
    std::vector<cv::Mat> hsv_planes;
    cv::split(img_HSV, hsv_planes);
    cv::Mat img_h = hsv_planes[0];
    cv::Mat img_s = hsv_planes[1];
    cv::Mat img_v = hsv_planes[2];

    // HSV范围筛选
    cv::Mat mask_h, mask_s, mask_v;
    cv::inRange(img_h, 150, 180, mask_h);
    cv::inRange(img_s, 50, 255, mask_s);
    cv::inRange(img_v, 50, 255, mask_v);

    // 掩膜合并
    cv::Mat mask_h_and_s, mask;
    cv::bitwise_and(mask_h, mask_s, mask_h_and_s);
    cv::bitwise_and(mask_h_and_s, mask_v, mask);

    // 应用掩膜
    cv::Mat img_output;
    cv::bitwise_and(img_BGR, img_BGR, img_output, mask);

    // 显示结果
    cv::imshow("img", img_output);
    cv::waitKey(-1);
    cv::destroyAllWindows();

    return 0;
}