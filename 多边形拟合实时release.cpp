#include<opencv2/opencv.hpp>
#include<vector>

int main() {
    cv::VideoCapture img("C:/Users/16968/Desktop/123/test.mp4");
    cv::Mat frame;
    while (img.isOpened()) {
        img.read(frame);
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(18,20, 50), cv::Scalar(30, 255, 255), hsv);
        cv::Mat kernel=cv::Mat::ones(15,15,CV_8U);
        cv::morphologyEx(hsv,hsv,cv::MORPH_CLOSE,kernel);
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(hsv,contours,hierarchy,cv::RETR_LIST,cv::CHAIN_APPROX_SIMPLE);
        cv::Mat first=frame.clone();
        cv::drawContours(first,contours,-1,cv::Scalar(255,0,0),3);
        std::vector<cv::Point> approx;
        for(int i=0;i<contours.size();i++) {
            double epsilon=0.03*cv::arcLength(contours[i],true);
            cv::approxPolyDP(contours[i],approx,epsilon,true);
            cv::Rect rect = cv::boundingRect(approx);
            cv::rectangle(frame,rect,cv::Scalar(0,255,0),3);
        }
        cv::imshow("first",first);
        cv::imshow("image",frame);
        if(cv::waitKey(30) == 'q') {
            break;
        }
    }
    img.release();
    cv::destroyAllWindows();
    return 0;
}