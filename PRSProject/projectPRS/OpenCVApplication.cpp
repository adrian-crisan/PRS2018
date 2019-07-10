#include "stdafx.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2\text.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace cv::text;

int thresh = 50, N = 11;
const char* wndname = "Rectangle Detection";

static double angle(Point pt1, Point pt2, Point pt0) {
	
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

static void findSquares(const Mat& image, vector<vector<Point> >& squares) {

	squares.clear();

	Mat pyr, timg, gray0(image.size(), CV_8U), gray;
	
	//downscale and upscale the image to filter out the noise
	pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
	pyrUp(pyr, timg, image.size());
	vector<vector<Point>> contours;

	//find squares in every color plane of the image
	for (int c = 0; c < 3; c++) {
		int ch[] = { c,0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		for (int l = 0; l < N; l++) {
			//apply Canny to catch squares with gradient shading
			if (l == 0) {
				Canny(gray0, gray, 0, thresh, 5);
				//dilate Canny outputs to remove potential holes 
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else {
				gray = gray0 >= (l + 1) * 255 / N;
			}

			//find contours and store them as a list
			findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			vector<Point> approx;

			//test each contour
			for (size_t i = 0; i < contours.size(); i++) {
				//approx contour with accuracy proportional to the contour perim
				approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);

				if (approx.size() == 4 && fabs(contourArea(approx)) > 1000 && fabs(contourArea(approx)) < 10000 && isContourConvex(approx)) {
					double maxCosine = 0;
					for (int j = 2; j < 5; j++) {
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
					}
					if (maxCosine < 0.2) {
						squares.push_back(approx);
					}
				}
			}
		}
	}
}

static void drawSquares(Mat& image, const vector<vector<Point>>& squares) {

	for (size_t i = 0; i < 1; i++) {
		const Point* p = &squares[i][0];
		const Point* p2 = &squares[i][1];
		const Point* p3 = &squares[i][2];
		const Point* p4 = &squares[i][3];
		int xArr[4] = { p->x, p2->x, p3->x, p4->x };
		int yArr[4] = { p->y, p2->y, p3->y, p4->y };
		int xMax = -INT_MAX, yMax = INT_MAX, xMin = INT_MAX, yMin = -INT_MAX;
		for (int j = 0; j < 3; j++) {
			if (xArr[j] > xMax) {
				xMax = xArr[j];
			}
			if (xArr[j] < xMin) {
				xMin = xArr[j];
			}
			if (yArr[j] > yMin) {
				yMin = yArr[j];
			}
			if (yArr[j] < yMax) {
				yMax = yArr[j];
			}
		}
		cout << xMax << "  " << xMin << "  " << yMax << "  " << yMin << endl;
		int height = yMin - yMax;
		int width = xMax - xMin;
		
		Rect rec(xMin, yMax, width, height);
		Mat roi = image(rec);
		//imshow("plate", roi);
		imwrite("plate.bmp", roi);
		
		int n = (int)squares[i].size();
		polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 1, LINE_AA);
	}

	imshow(wndname, image);
	imwrite("detected.bmp", image);
	//waitKey(0);
}

class comparator {
public:
	bool operator()(vector<Point> c1, vector<Point>c2) {
		return boundingRect(Mat(c1)).x < boundingRect(Mat(c2)).x;
	}
};

void extractContours(Mat& image, vector<vector<Point>> contours_poly) {
	//sort contours by x value
	sort(contours_poly.begin(), contours_poly.end(), comparator());

	//extract all contours in a loop
	for (int i = 0; i < contours_poly.size(); i++) {
		Rect r = boundingRect(Mat(contours_poly[i]));
		//mask used to take only pixels inside the contour
		Mat mask = Mat::zeros(image.size(), CV_8UC1);
		drawContours(mask, contours_poly, i, Scalar(255), FILLED);
		//extract the character using the mask
		Mat extractPic;
		image.copyTo(extractPic, mask);
		Mat resizedPic = extractPic(r);
		Mat image = resizedPic.clone();
		imshow("image", image);

		stringstream searchMask;
		searchMask << i << ".bmp";
		resize(resizedPic, resizedPic, Size(28, 28), 0, 0, INTER_LINEAR);
		imwrite(searchMask.str(), resizedPic);
	}
}

void getContours(const char* filename) {

	Mat img = imread(filename, 0);
	Size size(3, 3);
	//apply blur to to smooth edges and use adaptive thresholding
	GaussianBlur(img, img, size, 0);
	adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 2);
	bitwise_not(img, img);
	Mat img2 = img.clone();
	//rotate the image to be horizontal
	vector<Point> points;
	Mat_<uchar>::iterator it = img.begin<uchar>();
	Mat_<uchar>::iterator end = img.end<uchar>();

	for (; it != end; ++it) 
		if (*it)
			points.push_back(it.pos());

	RotatedRect box = minAreaRect(Mat(points));
	double angle = box.angle;
	
	if (angle < -45.)
		angle += 90.;
	
	Point2f vertices[4];
	box.points(vertices);
	for (int i = 0; i < 4; i++) 
		line(img, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 1, LINE_AA);
	
	Mat rot_mat = getRotationMatrix2D(box.center, angle, 1);
	Mat rotated;
	warpAffine(img2, rotated, rot_mat, img.size(), INTER_CUBIC);
	Size box_size = box.size;
	if (box.angle < -45.)
		swap(box_size.width, box_size.height);
	Mat cropped;
	imwrite("rotated.bmp", rotated);
	getRectSubPix(rotated, box_size, box.center, cropped);
	imshow("cropped", cropped);
	imwrite("ex.bmp", cropped);

	//make image clones to show
	Mat cropped2 = cropped.clone();
	cvtColor(cropped2, cropped2, COLOR_GRAY2RGB);

	Mat cropped3 = cropped.clone();
	cvtColor(cropped3, cropped3, COLOR_GRAY2RGB);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	//find contours
	findContours(cropped, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS, Point(0, 0));
	//approximate and get polygonal contours and bounding rectangles and circles
	vector<vector<Point>> contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f> center(contours.size());
	vector<float> radius(contours.size());

	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
	}

	//get only important contours
	vector<vector<Point>> validContours;
	for (int i = 0; i < contours_poly.size(); i++) {
		Rect r = boundingRect(Mat(contours_poly[i]));
		if (r.area() < 50) continue;
		bool inside = false;
		for (int j = 0; j < contours_poly.size(); j++) {
			if (j == i)
				continue;
			Rect r2 = boundingRect(Mat(contours_poly[j]));
			if (r2.area() < 50 || r2.area() < r.area())
				continue;
			if (r.x > r2.x && r.x + r.width < r2.x + r2.width &&
				r.y > r2.y && r.y + r.height < r2.y + r2.height) {
				inside = true;
			}
		}
		if (inside) continue;
		validContours.push_back(contours_poly[i]);
	}
	//get bounding rectangles
	for (int i = 0; i < validContours.size(); i++) {
		boundRect[i] = boundingRect(Mat(validContours[i]));
	}

	Scalar color = Scalar(0, 255, 0);
	for (int i = 0; i < validContours.size(); i++) {
		if (boundRect[i].area() < 100) continue;
		drawContours(cropped2, validContours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(cropped2, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
	}

	imshow("contours", cropped2);
	extractContours(cropped3, validContours);
	//waitKey(0);
}

Mat binarization(Mat img) {
	Mat bin(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			bin.at<uchar>(i, j) = 0;
		}
	}
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) > 128) {
				bin.at<uchar>(i, j) = 255;
			}
		}
	}
	return bin;
}

void bayes(Mat testImg) {

	const int nrclasses = 10;
	char fname[256];
	int index = 0;
	int rowX = 0, k = 0;
	Mat X(500, 784, CV_8UC1);
	Mat Y(500, 1, CV_8UC1);

	Mat priors(nrclasses, 1, CV_64FC1);
	const int d = 28 * 28;
	Mat likelihood(nrclasses, d, CV_64FC1);
	int elem[5];

	for (int c = 0; c < 10; c++) {
		index = 0;
		while (index < 50) {
			sprintf(fname, "bayes/train/%d/%06d.png", c, index);
			Mat img = imread(fname, 0);
			if (img.cols == 0) {
				break;
			}
			Mat bin = binarization(img);
			for (int i = 0; i < bin.rows; i++) {
				for (int j = 0; j < bin.cols; j++) {
					X.at<uchar>(rowX, k) = bin.at<uchar>(i, j);
					k++;
				}
			}
			Y.at<uchar>(rowX) = c;
			k = 0;
			index++;
			rowX++;
		}
		elem[c] = index;
	}

	for (int i = 0; i < nrclasses; i++) {
		priors.at<double>(i) = 50.0 / 500;
	}
	for (int i = 0; i < nrclasses; i++) {
		for (int j = 0; j < d; j++) {
			likelihood.at<double>(i, j) = 0.0;
		}
	}
	for (int i = 0; i < 500; i++) {
		for (int j = 0; j < d; j++) {
			if (X.at<uchar>(i, j) == 255) {
				likelihood.at<double>((int)Y.at<uchar>(i), j) += 1.0;
			}
		}
	}

	for (int i = 0; i < nrclasses; i++) {
		for (int j = 0; j < d; j++) {
			double val = likelihood.at<double>(i, j) + 1.0;
			likelihood.at<double>(i, j) = (val / (double)(nrclasses + elem[i]));
		}
	}

	fstream wfile;
	wfile.open("bayes/likelihood.txt");
	for (int i = 0; i < nrclasses; i++) {
		for (int j = 0; j < d; j++) {
			wfile << likelihood.at<double>(i, j) << "  ";
		}
		wfile << std::endl;
	}

	double p[nrclasses];
	int count = 0;
	
	Mat binTestImg = binarization(testImg);
	for (int i = 0; i < nrclasses; i++) {
		p[i] = log(priors.at<double>(i));
		count = 0;
		for (int j = 0; j < 28; j++) {
			for (int k = 0; k < 28; k++) {
				if (binTestImg.at<uchar>(j, k) == 255) {
					p[i] += log(likelihood.at<double>(i, count));
				}
				else {
					p[i] += log(1.0 - likelihood.at<double>(i, count));
				}
				count++;
			}
		}
	}

	double max = -INT_MAX;
	int pred = -1;
	for (int i = 0; i < nrclasses; i++) {
		if (p[i] > max) {
			max = p[i];
			pred = i;
		}
	}

	std::cout << "Predicted class: " << pred;

	waitKey(0);
}

int main() {

	vector<vector<Point> > squares;
	string path = "images/car3.jpg";
	Mat image = imread(path, IMREAD_COLOR);
	if (image.empty()) {
		cout << "Couldn't load image." << endl;
	}
	
	findSquares(image, squares);
	drawSquares(image, squares);
	
	char filename[256] = "plate.bmp";

	getContours(filename);
	Mat testImg = imread("10.png", IMREAD_GRAYSCALE);

	bayes(testImg);

	return 0;
}