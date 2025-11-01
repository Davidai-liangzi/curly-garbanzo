#include<opencv2/opencv.hpp>
int main() {
    cv::Mat img=cv::imread("C:/Users/16968/Desktop/123/1280x1280.PNG");
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(95, 120,70), cv::Scalar(110, 255, 255), hsv);
    cv::medianBlur(hsv, hsv, 3);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(hsv,contours,hierarchy,cv::RETR_LIST,cv::CHAIN_APPROX_SIMPLE);
    cv::Mat first=img.clone();
    cv::drawContours(first,contours,-1,cv::Scalar(255,0,0),3);
    for(int i=0;i<contours.size();i++) {
        double area=cv::contourArea(contours[i]);
        if (area>250&&area<3000) {
            double epsilon=0.03*cv::arcLength(contours[i],true);
            std::vector<cv::Point> contour;
            cv::approxPolyDP(contours[i],contour,epsilon,true);
            if (contour.size()<=6) {
                cv::Rect rect=cv::boundingRect(contour);
                cv::rectangle(img,rect,cv::Scalar(255,0,0),3);
            }
        }
    }
    cv::imshow("Contours", img);
    cv::imshow("HSV", first);
    cv::waitKey();
    return 0;
}