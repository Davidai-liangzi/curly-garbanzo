#include <opencv2/opencv.hpp>
#include<vector>
int main() {

    cv::Mat img = cv::imread("C:/Users/16968/Desktop/123/1280x1280.PNG");

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);


    cv::Scalar low_blue(95,120, 70);
    cv::Scalar high_blue(110, 255, 255);

    // 创建蓝色掩膜
    cv::Mat blue_mask;
    cv::inRange(hsv, low_blue, high_blue, blue_mask);
    cv::medianBlur(blue_mask,blue_mask,3);
    cv::bitwise_not(blue_mask,blue_mask);

    cv::Mat blue_result;
    cv::bitwise_and(img, img,blue_result, blue_mask);
    std::vector<std::vector<cv::Point> > cont;
    std::vector<cv::Vec4i> l;
    cv::findContours(blue_mask,cont,l,cv::RETR_LIST,cv::CHAIN_APPROX_NONE);
    cv::Mat first=img.clone();
    cv::drawContours(first,cont,-1,(0,0,255),3);
    cv::Mat result=img.clone();
    for (int i=0;i<cont.size();i++) {
        double area=cv::contourArea(cont[i]);
        if (area>250 && area<3000) {
            // 正确计算 epsilon
            double epsilon = 0.03 * cv::arcLength(cont[i], true);

            // 正确使用 approxPolyDP
            std::vector<cv::Point> approx;
            cv::approxPolyDP(cont[i], approx, epsilon, true);

            if (approx.size() <= 6) {
                cv::Rect rect = cv::boundingRect(cont[i]);
                cv::rectangle(result, rect, cv::Scalar(0, 0, 255), 2);
            }
            cv::Rect rect=cv::boundingRect(cont[i]);
            cv::rectangle(result,rect,cv::Scalar(0,0,255),3);
        }
    }

    cv::imshow("contours", first);
    cv::imshow("result", result);

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}