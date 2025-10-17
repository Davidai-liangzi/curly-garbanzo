#include<opencv2/opencv.hpp>
#include<vector>
int main() {
    cv::VideoCapture img("C:/Users/16968/Desktop/123/test.mp4");
    while (img.isOpened()) {
        cv::Mat frame;
        bool ret=img.read(frame);

        cv::Mat hsv;
        cv::cvtColor(frame,hsv,cv::COLOR_BGR2HSV);

        cv::Scalar down=cv::Scalar(18,20,50);
        cv::Scalar up=cv::Scalar(30,255,255);
        cv::Mat mask;
        cv::inRange(hsv,down,up,mask);

        cv::Mat kernel=cv::Mat::ones(20,20,CV_8U);
        cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,kernel);
        cv::morphologyEx(mask,mask,cv::MORPH_OPEN,kernel);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(mask,contours,cv::RETR_LIST,cv::CHAIN_APPROX_NONE);

        for(int i=0;i<contours.size();i++) {
            if (contours[i].size()>10) {
                cv::Rect rect = cv::boundingRect(contours[i]);
                cv::rectangle(frame,rect,cv::Scalar(255,0,0),2);

                double epsilon=0.03*cv::arcLength(contours[i],true);
                std::vector<cv::Point> contourPoints;
                cv::approxPolyDP(contours[i],contourPoints,epsilon,true);

                std::vector<std::vector<cv::Point>>polys={contourPoints} ;


            }
        }

        cv::imshow("image",frame);
        if (cv::waitKey(33) == 'q') {
            break;
        }

    }
    img.release();
    cv::destroyAllWindows();
    return 0;
}