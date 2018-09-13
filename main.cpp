#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat openKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat closeKernel = getStructuringElement(MORPH_RECT, Size(3, 3));


void onMouse(int evt, int x, int y, int flags, void *param) {
    if (evt == CV_EVENT_LBUTTONDOWN) {
        std::vector<cv::Point2f> *ptPtr = (std::vector<cv::Point2f> *) param;
        ptPtr->push_back(Point2f(x, y));
    }
}

Mat getCoordinates(vector<Point2f> ob, const Mat rotation_matrix, const Mat translation_matrix, const Mat camera_matrix) {

    Mat object = Mat::ones(3, 1, cv::DataType<double>::type);
    object.at<double>(0, 0) = ob[0].x;
    object.at<double>(1, 0) = ob[0].y;

    Mat tempMat, tempMat2;
    double s;
    tempMat = rotation_matrix.inv() * camera_matrix.inv() * object;
    tempMat2 = rotation_matrix.inv() * translation_matrix;
    s = tempMat2.at<double>(2, 0);
    s /= tempMat.at<double>(2, 0);
    return rotation_matrix.inv() * (s * camera_matrix.inv() * object - translation_matrix);
}

float getInitialHeight(const vector<float> &heights) {
    vector<float> tempHeights(heights.begin(), heights.end());
    sort(tempHeights.begin(), tempHeights.end());
    vector<int> trimmed_heights(tempHeights.begin() + 50, tempHeights.end() - 50);
    int mid = 399 / 2;
    float h_med = (trimmed_heights[mid] + trimmed_heights[mid + 1]) / 2;
    float num(0), den(0);
    for (auto i:trimmed_heights) {
        num += (1 - (abs(i - h_med) / h_med)) * i;
        den += (1 - (abs(i - h_med) / h_med));
    }
    return num / den;
}

float getHeight(float h_old, float h_new) {
    float num = (1 - abs(h_old - h_new) / h_old) * h_new + h_old;
    float den = 1 + (1 - abs(h_old - h_new) / h_old);
    return num / den;
}

double getHeightMean(const vector<float> &heights) {
    int size = heights.size();
    size *= 0.1;
    vector<float> tempHeights(heights.begin(), heights.end());
    sort(tempHeights.begin(), tempHeights.end());
    vector<float> trimmedHeights(tempHeights.begin() + size, tempHeights.end() - size);
    double meanHeight(0.0);
    for (auto i:trimmedHeights) {
        meanHeight += i;
    }
    return meanHeight / trimmedHeights.size();
}

tuple<Mat, vector<Point2f>, float> backSegment(Mat frame, const Mat background) {

    Mat frameGray = frame.clone();
    Mat backGray = background.clone();
    cvtColor(frameGray, frameGray, CV_BGR2GRAY);
    cvtColor(backGray, backGray, CV_BGR2GRAY);
    GaussianBlur(frameGray, frameGray, Size(13, 13), 0);
    GaussianBlur(backGray, backGray, Size(13, 13), 0);
    Mat delta, thresh;
    absdiff(frameGray, backGray, delta);
    threshold(delta, thresh, 15, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
    Mat fgmask, fgcopy;
    fgmask = thresh.clone();
    fgcopy = thresh.clone();
    morphologyEx(fgmask, fgmask, MORPH_OPEN, openKernel, Point(0, 0));
    morphologyEx(fgmask, fgmask, MORPH_CLOSE, closeKernel, Point(0, 0), 5);

    for (int i = 0; i < fgmask.cols; i++) {
        if (fgmask.at<char>(0, i) == 0) {
            floodFill(fgmask, cv::Point(i, 0), 255, 0, 10, 10);
        }
        if (fgmask.at<char>(fgmask.rows - 1, i) == 0) {
            floodFill(fgmask, cv::Point(i, fgmask.rows - 1), 255, 0, 10, 10);
        }
    }

    for (int i = 0; i < fgmask.rows; i++) {
        if (fgmask.at<char>(i, 0) == 0) {
            floodFill(fgmask, cv::Point(0, i), 255, 0, 10, 10);
        }
        if (fgmask.at<char>(i, fgmask.cols - 1) == 0) {
            floodFill(fgmask, cv::Point(fgmask.cols - 1, i), 255, 0, 10, 10);
        }
    }


    for (int row = 0; row < fgmask.rows; ++row) {
        for (int col = 0; col < fgmask.cols; ++col) {
            if (fgmask.at<char>(row, col) == 0) {
                fgcopy.at<char>(row, col) = 255;
            }
        }
    }

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Point2f> uv(1);

    findContours(fgcopy, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<Rect> boundRect(contours.size());

    for (int i = 0; i < contours.size(); i++) {
        boundRect[i] = boundingRect(Mat(contours[i]));
    }
    int min_x(boundRect[0].x), min_y(boundRect[0].y), max_x(boundRect[0].x + boundRect[0].width), max_y(
            boundRect[0].y + boundRect[0].height);
    for (int i = 1; i < contours.size(); i++) {
        if (boundRect[i].x < min_x) {
            min_x = boundRect[i].x;
        }
        if (boundRect[i].y < min_y) {
            min_y = boundRect[i].y;
        }
    }
    for (int i = 1; i < contours.size(); i++) {
        if (boundRect[i].x + boundRect[i].width > max_x) {
            max_x = boundRect[i].x + boundRect[i].width;
        }
        if (boundRect[i].y + boundRect[i].height > max_y) {
            max_y = boundRect[i].y + boundRect[i].height;
        }
    }
    if ((max_y - min_y) * (max_x - min_x) >= (640 * 480) * 0.60) {
        Mat emptyFrame;
        return tuple<Mat, vector<Point2f>, float>{emptyFrame, uv, 0.0};
    } else {
        rectangle(frame, Point(min_x, min_y), Point(max_x, max_y), Scalar(0, 0, 255), 5, 8, 0);
        uv[0] = Point((max_x + min_x) / 2, max_y);
        float height = max_y - min_y;
        return tuple<Mat, vector<Point2f>, float>{frame, uv, height};
    }
}


tuple<float, float> getLineEquation(Vec4f line) {
    float m = line[1] / line[0];
    float c = m * (-line[2]) + line[3];
    return tuple<float, float>(m, c);
}

int main(int argc, char **argv) {
    VideoCapture cap;

    if (!cap.open(0)) {
        cout << "No web cam found!";
        return 0;
    }

    FileStorage fs("out_camera_data.yml", FileStorage::READ);

    Mat camera_matrix, distortion_matrix;
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> distortion_matrix;
    fs.release();

    Mat frame;
    vector<Point2f> imagePoints;
    vector<Point2f> heightPoints1;
    vector<Point2f> heightPoints2;
    vector<Point2f> heightPoints3;

    cap >> frame;
    imwrite("background.jpg",frame);

    Mat bg = imread("background.jpg", 1);
    Mat imageFrame = bg.clone();
    Mat heightFrame1 = bg.clone();
    Mat heightFrame2 = bg.clone();
    Mat heightFrame3 = bg.clone();
    Mat background = bg.clone();

    namedWindow("Select 4 image points");
    cvSetMouseCallback("Select 4 image points", onMouse, (void *) &imagePoints);

    while (true) {
        putText(imageFrame, "Select 4 image points", Point(5, 15), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(255, 0, 0),
                1, CV_AA);
        imshow("Select 4 image points", imageFrame);

        int k = waitKey() & 255;
        if (k == 27) {
            break;
        }
    }

    destroyWindow("Select 4 image points");

    vector<Point3f> worldPoints;
    worldPoints.emplace_back(Point3f(0., 0., 0.));
    worldPoints.emplace_back(Point3f(2400., 300., 0.));
    worldPoints.emplace_back(Point3f(1800., 600., 0.));
    worldPoints.emplace_back(Point3f(300., 900., 0.));

    Mat r(1, 3, DataType<double>::type);
    Mat t(1, 3, DataType<double>::type);
    Mat rotation_matrix(3, 3, DataType<double>::type);

    solvePnP(worldPoints, imagePoints, camera_matrix, distortion_matrix, r, t);
    Rodrigues(r, rotation_matrix);

    namedWindow("Select height points of object 1");
    cvSetMouseCallback("Select height points of object 1", onMouse, (void *) &heightPoints1);

    while (true) {
        putText(heightFrame1, "Select height points of object 1", Point(5, 15), FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                Scalar(255, 0, 0), 1, CV_AA);
        imshow("Select height points of object 1", heightFrame1);

        int k = waitKey() & 255;
        if (k == 27) {
            break;
        }
    }

    destroyWindow("Select height points of object 1");

    float height1 = 652.0;
    float pxHeight1 = abs(heightPoints1[0].y - heightPoints1[1].y);

    namedWindow("Select height points of object 2");
    cvSetMouseCallback("Select height points of object 2", onMouse, (void *) &heightPoints2);

    while (true) {
        putText(heightFrame2, "Select height points of object 2", Point(5, 15), FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                Scalar(255, 0, 0), 1, CV_AA);
        imshow("Select height points of object 2", heightFrame2);

        int k = waitKey() & 255;
        if (k == 27) {
            break;
        }
    }

    destroyWindow("Select height points of object 2");

    float height2 = 655.0;
    float pxHeight2 = abs(heightPoints2[0].y - heightPoints2[1].y);

    namedWindow("Select height points of object 3");
    cvSetMouseCallback("Select height points of object 3", onMouse, (void *) &heightPoints3);

    while (true) {
        putText(heightFrame3, "Select height points of object 3", Point(5, 15), FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                Scalar(255, 0, 0), 1, CV_AA);
        imshow("Select height points of object 3", heightFrame3);

        int k = waitKey() & 255;
        if (k == 27) {
            break;
        }
    }

    destroyWindow("Select height points of object 3");

    float height3 = 658.0;
    float pxHeight3 = abs(heightPoints3[0].y - heightPoints3[1].y);

    undistortPoints(heightPoints1, heightPoints1, camera_matrix, distortion_matrix, cv::noArray(), camera_matrix);
    undistortPoints(heightPoints2, heightPoints2, camera_matrix, distortion_matrix, cv::noArray(), camera_matrix);
    undistortPoints(heightPoints3, heightPoints3, camera_matrix, distortion_matrix, cv::noArray(), camera_matrix);

    heightPoints1.erase(heightPoints1.begin());
    heightPoints2.erase(heightPoints2.begin());
    heightPoints3.erase(heightPoints3.begin());

    float distance1 = 300;
    float mmpxRatio1 = height1 / pxHeight1;

    float distance2 = 600;
    float mmpxRatio2 = height2 / pxHeight2;

    float distance3 = 900;
    float mmpxRatio3 = height3 / pxHeight3;

    vector<Point2f> heightCoordinates;
    heightCoordinates.emplace_back(Point2f(distance1, mmpxRatio1));
    heightCoordinates.emplace_back(Point2f(distance2, mmpxRatio2));
    heightCoordinates.emplace_back(Point2f(distance3, mmpxRatio3));

    Vec4f line;
    fitLine(heightCoordinates, line, CV_DIST_L2, 0, 0.01, 0.01);
    float m, c;
    tie(m, c) = getLineEquation(line);

    vector<float> heights;

    Mat openKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat closeKernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    cap >> frame;
    background = frame.clone();
    double meanHeight(0.0);
    while (true) {
        cap >> frame;
        Mat detectionFrame;
        vector<Point2f> uv;
        vector<Point2f> ob;
        float obpxHeight;
        Mat fgmask;
        tie(detectionFrame, uv, obpxHeight) = backSegment(frame, background);

        if (detectionFrame.empty()) {
            detectionFrame = frame;
            heights.clear();
            putText(detectionFrame, "NO HUMAN FOUND", Point(10, 30), FONT_HERSHEY_COMPLEX_SMALL, 1.5, Scalar(255, 0, 0),
                    1, CV_AA);
        } else {
            undistortPoints(uv, ob, camera_matrix, distortion_matrix, cv::noArray(), camera_matrix);
            Mat coordinates = getCoordinates(ob, rotation_matrix, t, camera_matrix);
            float obDist = coordinates.at<double>(1, 0);
            float obRatio = m * obDist + c;
            float obHeight = obpxHeight * obRatio;
            putText(detectionFrame, "X: " + to_string(coordinates.at<double>(0, 0)), Point(10, 30),
                    FONT_HERSHEY_COMPLEX_SMALL, 1.4, Scalar(255, 0, 0), 1, CV_AA);
            putText(detectionFrame, "Y: " + to_string(coordinates.at<double>(1, 0)), Point(10, 55),
                    FONT_HERSHEY_COMPLEX_SMALL, 1.4, Scalar(255, 0, 0), 1, CV_AA);
            if (heights.size() < 500) {
                heights.push_back(obHeight);
                putText(detectionFrame, "Height: (Acquiring data)", Point(10, 80), FONT_HERSHEY_COMPLEX_SMALL, 1.4,
                        Scalar(255, 0, 0), 1, CV_AA);
            } else {
                if (heights.size() == 10000) {
                    putText(detectionFrame, "Height: " + to_string(meanHeight), Point(15, 80),
                            FONT_HERSHEY_COMPLEX_SMALL, 1.4, Scalar(255, 0, 0), 1, CV_AA);
                } else {
                    heights.push_back(obHeight);
                    meanHeight = getHeightMean(heights);
                    putText(detectionFrame, "Height: " + to_string(meanHeight), Point(15, 80),
                            FONT_HERSHEY_COMPLEX_SMALL, 1.4, Scalar(255, 0, 0), 1, CV_AA);
                }
            }
        }
        imshow("Frame", detectionFrame);
        if (waitKey(1) == 27) {
            break;
        }
    }
    return 0;
}