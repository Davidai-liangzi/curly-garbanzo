#include<opencv2/opencv.hpp>
#include<vector>
#include<algorithm>

int main() {
    cv::VideoCapture img(0);
    img.set(cv::CAP_PROP_FRAME_WIDTH,1280);
    img.set(cv::CAP_PROP_FRAME_HEIGHT,720);
    cv::Mat frame;
    cv::Mat camera_matrix=(cv::Mat_<double>(3,3) <<
        969.1364,0,616.8481,
        0,970.6934,389.1486,
        0,0,1);
    cv::Mat dist_coeff=cv::Mat::zeros(4,1,CV_64F);

    std::vector<cv::Point3f> obj_points;
    obj_points.push_back(cv::Point3f(-0.105,-0.1485,0));
    obj_points.push_back(cv::Point3f(0.105,-0.1485,0));
    obj_points.push_back(cv::Point3f(0.105,0.1485,0));
    obj_points.push_back(cv::Point3f(-0.105,0.1485,0));

    std::vector<cv::Point3f> axispoints3D;
    axispoints3D.push_back(cv::Point3f(0,0,0));
    axispoints3D.push_back(cv::Point3f(0.1,0,0));
    axispoints3D.push_back(cv::Point3f(0,0.1,0));
    axispoints3D.push_back(cv::Point3f(0,0,0.1));

    while (img.isOpened()) {
        img.read(frame);

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, gray, 60, 255, cv::THRESH_BINARY);
        // cv::inRange(gray,0,60,gray);
        //cv::threshold(gray, gray, 100, 255, cv::THREH_BINARY);
        //cv::GaussianBlur(gray, gray, cv::Size(1,1), 0);
        cv::Mat kernel=cv::Mat::ones(10,10,CV_8U);
        //cv::morphologyEx(gray,gray,cv::MORPH_CLOSE,kernel);
        //cv::morphologyEx(gray,gray,cv::MORPH_OPEN,kernel);
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(gray,contours,cv::RETR_LIST,cv::CHAIN_APPROX_SIMPLE);
        for (int i=0;i<contours.size();i++) {
            double area=cv::contourArea(contours[i]);
            if (area>10000&&area<200000) {
                double epsilon=0.03*cv::arcLength(contours[i],true);
                std::vector<cv::Point> approx;
                cv::approxPolyDP(contours[i],approx,epsilon,true);
                if (approx.size()==4) {
                    std::vector<cv::Point2f> approx2D;
                    for (int a=0;a<4;a++) {
                        cv::line(frame,approx[a],approx[(a+1)%4],cv::Scalar(0,255,0),3);
                        approx2D.push_back(cv::Point2f(approx[a].x,approx[a].y));
                    }
                    cv::Mat rvec,tvec;
                    std::vector<cv::Point2f> axispoints2D;
                    cv::solvePnP(obj_points,approx2D,camera_matrix,dist_coeff,rvec,tvec);
                    cv::projectPoints(axispoints3D,rvec,tvec,camera_matrix,dist_coeff,axispoints2D);

                    cv::Point2f center=axispoints2D[0];
                    cv::arrowedLine(frame,center,axispoints2D[1],cv::Scalar(255,0,0),3);
                    cv::arrowedLine(frame,center,axispoints2D[2],cv::Scalar(0,255,0),3);
                    cv::arrowedLine(frame,center,axispoints2D[3],cv::Scalar(0,0,255),3);

                    cv::putText(frame,"X",axispoints2D[1],cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(255,0,0),3);
                    cv::putText(frame,"Y",axispoints2D[2],cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(0,255,0),3);
                    cv::putText(frame,"Z",axispoints2D[3],cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(0,0,255),3);


                }
            }
        }
        cv::imshow("test",gray);
        cv::imshow("image",frame);
        cv::waitKey(1);
    }
    img.release();
    cv::destroyAllWindows();
    return 0;
}