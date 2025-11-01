#include<opencv2/opencv.hpp>

int main() {
    cv::Mat img=cv::imread("C:/Users/16968/Desktop/123/bed_pic.png");
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv,cv::Scalar(150,50,50),cv::Scalar(180,255,255),hsv);
    cv::Mat result;
    cv::bitwise_and(img,img,result,hsv);
    cv::imshow("result",result);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}