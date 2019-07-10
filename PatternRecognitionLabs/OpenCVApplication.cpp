// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <time.h>
#include <fstream>
#include <random>

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


//LAB1
void leastMean() {
	FILE* f = fopen("Images/points_LeastSquares/points0.txt", "r");
	int n = 0;
	fscanf(f, "%d\n", &n);
	
	Point2d points[500];
	float x, y;
	fscanf(f, "%f   %f\n", &x, &y);
	points[0] = Point2d(x, y);

	float maxX = x, maxY = y;
	float minX = x, minY = y;

	//citire puncte
	for (int i = 1; i < n; i++) {
		fscanf(f, "%f    %f\n", &x, &y);
		points[i] = Point2d(x, y);
		if (maxX < x) maxX = x;
		if (maxY < y) maxY = y;
		if (minX > x) minX = x;
		if (minY > y) minY = y;
	}

	//translatare
	for (int i = 0; i < n; i++) {
		points[i].x -= minX;
		points[i].y -= minY;
	}

	printf("Nr = %d, MaxX = %f, MaxY = %f\n", n, maxX, maxY);

	//creare imagine alba
	Mat img(500, 500, CV_8UC3);
	printf("Ajung1\n");
	for (int i = 0; i < 500; i++) {
		for (int j = 0; j < 500; j++) {
			img.at<Vec3b>(i, j)[0] = 255;
			img.at<Vec3b>(i, j)[1] = 255;
			img.at<Vec3b>(i, j)[2] = 255;
		}
	}

	printf("Ajung2\n");

	//innegrire puncte
	for (int i = 0; i < n; i++) {
		img.at<Vec3b>(points[i].y, points[i].x)[0] = 0;
		img.at<Vec3b>(points[i].y, points[i].x)[1] = 0;
		img.at<Vec3b>(points[i].y, points[i].x)[2] = 0;
	}

	printf("Ajung3\n");


	//MODEL 1
	float teta0 = 0.0f, teta1 = 0.0f;
	float sumProd = 0.0f, sumXi = 0.0f, sumYi = 0.0f, sumXiSquare = 0.0f;
	float sumDiffSquares = 0.0f;

	for (int i = 0; i < n; i++){
		sumProd += points[i].x * points[i].y;
		sumXi += points[i].x;
		sumYi += points[i].y;
		sumXiSquare += points[i].x * points[i].x;
		//Model2
		sumDiffSquares += points[i].y * points[i].y - points[i].x * points[i].x;
	}

	teta1 = (n * sumProd - sumXi * sumYi) / (n * sumXiSquare - sumXi * sumXi);
	teta0 = (1.0f / n) * (sumYi - teta1 * sumXi);

	float fi = atan(teta1);
	printf("Fi = %f, TETA0= %f, Teta1= %f\n", fi,teta0, teta1);

	float valY1 = teta0 + teta1 * 0;
	float valY2 = teta0 + teta1 * maxX;

	line(img, Point(0, valY1), Point(maxX, valY2), Scalar(0, 0, 255), 4);

	//MODEL 2
	
	float beta = -(1.0f / 2.0f) * atan2(2 * sumProd - (2.0f / n) * sumXi * sumYi, sumDiffSquares + (1.0f / n) * sumXi * sumXi - (1.0f / n) * sumYi * sumYi);

	float ro = (1.0f / n) * (cos(beta) * sumXi + sin(beta) * sumYi);

	printf("Beta = %f, Ro = %f\n", beta, ro);

	valY1 = (ro - 0 * cos(beta))/ (sin(beta));
	valY2 = (ro - maxX * cos(beta)) / (sin(beta));

	line(img, Point(0, valY1), Point(maxX, valY2), Scalar(0, 255, 0),2);

	imshow("Points", img);
	waitKey();
}

//lab2
void ransac() {

	Mat img = imread("ransac/points3.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	srand(time(NULL));
	int height = img.rows;
	int width = img.cols;
	Point2d points[100];
	int k = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0) {
				points[k].x = j;
				points[k].y = i;
				k++;
			}
		}
	}
	double p = 0.99;
	double t = 10.0;
	double q = 0.8;
	double N = log(1.0 - p) / log(1.0 - q*q);
	int T = q * k;
	
	Point2d pointOne, pointTwo;
	int inliers = 0;
	int maxInliers = 0;
	double maxInliersA = 0, maxInliersB = 0, maxInliersC = 0;

	for (int i = 0; i < N; i++) {
		inliers = 0;
		int pointOnePos = rand() % k;
		int pointTwoPos = rand() % k;
		while (pointOnePos == pointTwoPos) {
			pointTwoPos = rand() % k;
		}
		pointOne = points[pointOnePos];
		pointTwo = points[pointTwoPos];

		double a = pointOne.y - pointTwo.y;
		double b = pointTwo.x - pointOne.x;
		double c = pointOne.x*pointTwo.y - pointTwo.x*pointOne.y;

		for (int i = 0; i < k; i++) {
			double dist = fabs(a*points[i].x + b*points[i].y + c) / sqrt(a*a + b*b);
			if (dist <= t) {
				inliers++;
			}
		}
		printf("%d ", inliers);
		if (inliers > maxInliers) {
			maxInliers = inliers;
			maxInliersA = a;
			maxInliersB = b;
			maxInliersC = c;
		}
		if (inliers > T) {
			Point2d one(0.0, -c / b);
			Point2d two(width, (-c - a*width) / b);
			line(img, one, two, Scalar(0, 0, 0));
			break;
		}

	}

	imshow("img", img);
	waitKey(0);
}

//lab3
struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

void houghTransform() {
	Mat img = imread("hough/edge_simple.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat color = imread("hough/edge_simple.bmp", CV_LOAD_IMAGE_COLOR);
	int d = sqrt(img.cols*img.cols + img.rows*img.rows);
	Mat hough(360, d + 1, CV_32SC1, Scalar(0));
	int deltaT = 1;
	int deltaR = 1;
	int ro = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 255) {
				for (int t = 0; t < 360; t += deltaT) {
					ro = i*sin(t*PI/180.0) + j*cos(t*PI/180.0);
					if (ro >= 0 && ro <= 144) {
						hough.at<int>(t, ro)++;
					}
				}
			}
		}
	}
	double min, max;
	minMaxLoc(hough, &min, &max);
	Mat houghImg;
	hough.convertTo(houghImg, CV_8UC1, 255.f / (float)max);
	imshow("hough", houghImg);
	
	peak peaks[10000];
	int peaksNo = 0;
	bool checkMax;
	for (int t = 3; t < 357; t++) {
		for (int ro = 3; ro < d - 3; ro++) {
			checkMax = true;
			for (int i = t - 0; i <= t + 3; i++) {
				for (int j = ro - 3; j <= ro + 3; j++) {
					if (hough.at<int>(t,ro) == 0 || hough.at<int>(i, j) > hough.at<int>(t, ro)) {
						checkMax = false;
					}
				}
			}
			if (checkMax == true) {
				peaks[peaksNo].theta = t;
				peaks[peaksNo].ro = ro;
				peaks[peaksNo].hval = hough.at<int>(t, ro);
				peaksNo++;
			}
		}
	}
	std::cout << peaksNo << std::endl;
	std::sort(peaks, peaks + peaksNo);
	for (int i = 0; i < 10; i++) {
		std::cout << peaks[i].hval << std::endl;
		Point2d pt1, pt2;
		double x0 = peaks[i].ro * cos(peaks[i].theta*PI/180.0);
		double y0 = peaks[i].ro * sin(peaks[i].theta*PI/180.0);
		pt1.x = cvRound(x0 + 1000 * (-sin(peaks[i].theta*PI/180.0))); //??
		pt1.y = cvRound(y0 + 1000 * (cos(peaks[i].theta*PI/180.0))); //??
		pt2.x = cvRound(x0 - 1000 * (-sin(peaks[i].theta*PI/180.0))); //??
		pt2.y = cvRound(y0 - 1000 * (cos(peaks[i].theta*PI/180.0))); //??
		line(color, pt1, pt2, Scalar(255, 0, 0), 2);
	}
	imshow("img", color);
	waitKey(0);
}

void distanceTransform() {
	Mat img = imread("images_DT_PM/PatternMatching/unknown_object2.bmp", IMREAD_GRAYSCALE);
	Mat dt = img.clone();
	int di[8] = { -1,-1,-1,0,0,1,1,1 };
	int dj[8] = { -1,0,1,-1,1,-1,0,1 };
	int weight[8] = { 3, 2, 3, 2, 2, 3, 2, 3 };
	int min, val = 0;
	for (int i = 1; i < img.rows-1; i++) {
		for (int j = 1; j < img.cols-1; j++) {
			min = 10000;
			for (int k = 0; k < 4; k++) {
				val = dt.at<uchar>(i + di[k], j + dj[k]) + weight[k];
				if (val < min) {
					min = val;
				}
			}
			if (min < dt.at<uchar>(i, j)) {
				dt.at<uchar>(i, j) = min;
			}
		
		}
	}
	for (int i = img.rows - 2; i > 0; i--) {
		for (int j = img.cols - 2; j > 0; j--) {
			min = 10000;
			for (int k = 4; k < 8; k++) {
				val = dt.at<uchar>(i + di[k], j + dj[k]) + weight[k];
				if (val < min) {
					min = val;
				}
			}
			if (min < dt.at<uchar>(i, j)) {
				dt.at<uchar>(i, j) = min;
			}
		}
	}
	Mat temp = imread("images_DT_PM/PatternMatching/template.bmp", IMREAD_GRAYSCALE);
	int n = 0, s = 0;
	for (int i = 0; i < temp.rows; i++) {
		for (int j = 0; j < temp.cols; j++) {
			if (temp.at<uchar>(i, j) == 0) {
				n++;
				s += dt.at<uchar>(i, j);
			}
		}
	}
	double score = s / n;
	std::cout << "Matching score: " << score << std::endl;
	imshow("template", temp);
	imshow("DT", dt);
	waitKey(0);
}

void statistics() {
	char folder[256] = "faces";
	char fname[256];
	Mat I(400, 361, CV_8UC1);
	for (int k = 1; k <= 400; k++) {
		sprintf(fname, "%s/face%05d.bmp", folder, k);
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		int l = 0;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				I.at<uchar>(k-1, l) = img.at<uchar>(i, j);
				l++;
			}
		}
		l = 0;
	}

	int means[361];
	int s = 0;
	for (int i = 0; i < 361; i++) {
		for (int j = 0; j < 400; j++) {
			s += I.at<uchar>(j, i);
		}
		means[i] = (int)s / 400;
		s = 0;
	}
	std::ofstream meanValues;
	meanValues.open("meanValues.csv");
	for (int i = 0; i < 361; i++) {
		meanValues << means[i] << std::endl;
	}

	Mat cov(361, 361, CV_64FC1);
	double x;
	for (int i = 0; i < 361; i++) {
		for (int j = 0; j < 361; j++) {
			x = 0.0;
			for (int k = 0; k < 400; k++) {
				x += ((double)I.at<uchar>(k, i) - (double)means[i])*((double)I.at<uchar>(k, j) - (double)means[j]);
			}
			cov.at<double>(i, j) = x/400;
		}
	}
	std::ofstream covValues;
	covValues.open("covValues.csv");
	for (int i = 0; i < 361; i++) {
		for (int j = 0; j < 361; j++) {
			covValues << cov.at<double>(i, j) << std::endl;
		}
	}

	double stdDev[361];
	double val;
	for (int i = 0; i < 361; i++) {
		val = 0.0;
		for (int j = 0; j < 400; j++) {
			val += ((double)I.at<uchar>(j, i) - (double)means[i])*((double)I.at<uchar>(j, i) - (double)means[i]);
		}
		stdDev[i] = sqrt(val / 400);
	}

	Mat corMat(361, 361, CV_64FC1);
	for (int i = 0; i < 361; i++) {
		for (int j = 0; j < 361; j++) {
			corMat.at<double>(i, j) = cov.at<double>(i, j) / (stdDev[i] * stdDev[j]);
		}
	}
	std::ofstream corValues;
	corValues.open("correlationValues.csv");
	for (int i = 0; i < 361; i++) {
		for (int j = 0; j < 361; j++) {
			corValues << corMat.at<double>(i, j) << std::endl;
		}
	}

	Mat chart(256, 256, CV_8UC1);
	for (int k = 0; k < 400; k++) {
		chart.at<uchar>(I.at<uchar>(k,5*19+4), I.at<uchar>(k,5*19+14)) = 0;
	}
	imshow("chart", chart);
	std::cout << corMat.at<double>(5 * 19 + 4, 5 * 19 + 14);
	waitKey(0);
}

struct punct {
	Point p;
	int label;
};

void KmeansClustering(const int K) {
	Mat src = imread("kmeans/points4.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	std::default_random_engine gen;
	gen.seed(time(NULL));
	std::uniform_int_distribution<int> dist_img(0, 255);
	
	Vec3b *colors = (Vec3b*)malloc(K * sizeof(Vec3b));
	for (int i = 0; i < K; i++) {
		colors[i] = { (uchar)dist_img(gen), (uchar)dist_img(gen), (uchar)dist_img(gen) };
	}

	Point *m = (Point*)malloc(K * sizeof(Point));
	Point points[10000];
	int count = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {
				Point p(i, j);
				points[count++] = p;
			}
		}
	}

	std::uniform_int_distribution<int> dist_point(0, count);
	for (int i = 0; i < K; i++) {
		m[i] = points[(dist_point(gen))];
		std::cout << m[i] << std::endl;
	}


	double *distances = (double*)malloc(K * sizeof(double));
	punct *pointS = (punct*)malloc(count * sizeof(punct));
	Point *newCenters = (Point*)malloc(K * sizeof(Point));
	bool hasChanged = true;
	while(hasChanged) {
		for (int i = 0; i < count; i++) {
			for (int k = 0; k < K; k++) {
				distances[k] = ((m[k].x - points[i].x)*(m[k].x - points[i].x) + (m[k].y - points[i].y)*(m[k].y - points[i].y));
			}
			int min = 1000;
			int index = 0;
			for (int k = 0; k < K; k++) {
				if (distances[k] < min) {
					index = k;
					min = distances[k];
				}
			}
			pointS[i].label = index;
			pointS[i].p = points[i];
		}

		int *sumX = (int*)malloc(K * sizeof(int));
		int *sumY = (int*)malloc(K * sizeof(int));
		int *noPoints = (int*)malloc(K * sizeof(int));
		for (int i = 0; i < K; i++) {
			sumX[i] = 0;
			sumY[i] = 0;
			noPoints[i] = 0;
		}
		for (int i = 0; i < count; i++) {
			sumX[pointS[i].label] += pointS[i].p.x;
			sumY[pointS[i].label] += pointS[i].p.y;
			noPoints[pointS[i].label]++;
		}
		for (int k = 0; k < K; k++) {
			m[k].x = (int)(sumX[k] / noPoints[k]);
			m[k].y = (int)(sumY[k] / noPoints[k]);
		}
		hasChanged = false;
		for (int k = 0; k < K; k++) {
			if ((m[k].x != newCenters[k].x) || (m[k].y != newCenters[k].y)) {
				hasChanged = true;
			}
		}
		for (int k = 0; k < K; k++) {
			newCenters[k] = m[k];
		}
	}

	Mat voronoi = imread("kmeans/points4.bmp", CV_LOAD_IMAGE_COLOR);
	for (int i = 0; i < voronoi.rows; i++ ) {
		for (int j = 0; j < voronoi.cols; j++) {
			int min = 10000;
			int index = 0;
			for (int k = 0; k < K; k++) {
				distances[k] = sqrt(((m[k].x - i)*(m[k].x - i)) + ((m[k].y - j)*(m[k].y - j)));
				if (distances[k] < min) {
					index = k;
					min = distances[k];
				}
			}
			voronoi.at<Vec3b>(i, j) = colors[index];

		}
	}

	Mat colorSrc = imread("kmeans/points4.bmp", CV_LOAD_IMAGE_COLOR);
	for (int i = 0; i < colorSrc.rows; i++) {
		for (int j = 0; j < colorSrc.cols; j++) {
			if (colorSrc.at<Vec3b>(i, j)[0] != 0) {
				colorSrc.at<Vec3b>(i, j) = voronoi.at<Vec3b>(i, j);
			}
		}
	}

	imshow("voronoi", voronoi);
	imshow("voronoi + points", colorSrc);

	waitKey(0);
	getchar();
	getchar();

}

void pca() {

	std::fstream readFile;
	readFile.open("pca/pca2d.txt");
	int n = 0, d = 0;
	readFile >> n;
	readFile >> d;
	Mat X(n, d, CV_64FC1);
	for (int i = 0; i < X.rows; i++) {
		for (int j = 0; j < X.cols; j++) {
			readFile >> X.at<double>(i, j);
		}
	}

	double *means = (double*)malloc(d * sizeof(double));
	double s = 0;
	for (int i = 0; i < X.cols; i++) {
		for (int j = 0; j < X.rows; j++) {
			s += X.at<double>(j, i);
		}
		means[i] = s / n;
		s = 0;
	}

	for (int i = 0; i < X.rows; i++) {
		for (int j = 0; j < X.cols; j++) {
			X.at<double>(i, j) -= means[j];
		}
	}

	Mat C = (X.t() * X) / (n - 1);
	Mat Lambda, Q;
	eigen(C, Lambda, Q);
	Q = Q.t();

	for (int i = 0; i < d; i++) {
		std::cout << Lambda.at<double>(i) << std::endl;
	}

	int k = 2;
	Mat Qk(d, k, CV_64FC1);
	for (int i = 0; i < d; i++) {
		for (int j = 0; j < k; j++) {
			Qk.at<double>(i, j) = Q.at<double>(i, j);
		}
	}

	Mat Xcoef;
	Xcoef = X * Qk;
	Mat Qkt = Qk.t();
	Mat Xk = Xcoef * Qkt;

	
	double min1 = 0.0, min2 = 0.0, max1 = 0.0, max2 = 0.0;
	for (int i = 0; i < n; i++) {
		if (Xcoef.at<double>(i, 0) < min1) {
			min1 = Xcoef.at<double>(i, 0);
		}
		if (Xcoef.at<double>(i, 1) < min2) {
			min2 = Xcoef.at<double>(i, 1);
		}
		if (Xcoef.at<double>(i, 0) > max1) {
			max1 = Xcoef.at<double>(i, 0);
		}
		if (Xcoef.at<double>(i, 1) > max2) {
			max2 = Xcoef.at<double>(i, 1);
		}
	}

	int height = (int)(max2 - min2);
	int width = (int)(max1 - min1);
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = 255;
		}
	}
	for (int i = 0; i < n; i++) {
		int a = (int)(Xcoef.at<double>(i, 1)) - (int)min2;
		int b = (int)(Xcoef.at<double>(i, 0)) - (int)min1;
		img.at<uchar>(a, b) = 0;
	}

	Mat mao(n, d, CV_64FC1);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < d; j++) {
			mao.at<double>(i, j) = abs(X.at<double>(i, j) - Xk.at<double>(i, j));
		}
	}
	double sum = 0.0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < d; j++) {
			sum += mao.at<double>(i, j);
		}
	}
	double avg = sum / (n*d);

	std::cout << "Mins: " << min1 << "   " << min2 << std::endl;
	std::cout << "Maxs: " << max1 << "   " << max2 << std::endl;
	std::cout << "Mean absolute difference: " << avg << std::endl;

	imshow("img", img);
	waitKey(0);
	getchar();
	getchar();
}

int* computeHistogram(Mat img, int m) {

	int *hist = (int*)malloc(3 * m * sizeof(int));
	for (int i = 0; i < m * 3; i++) {
		hist[i] = 0;
	}
	int binWidth = 256 / m;
	int red, green, blue;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			red = img.at<Vec3b>(i, j)[2];
			green = img.at<Vec3b>(i, j)[1];
			blue = img.at<Vec3b>(i, j)[0];
			hist[red / binWidth]++;
			hist[m + green / binWidth]++;
			hist[m * 2 + blue / binWidth]++;
		}
	}
	return hist;
}

int compare(const void * a, const void * b)
{
	return (*(float*)a - *(float*)b);
}

void kNN() {

	int m = 8, k = 7;
	const int nrclasses = 6;
	char classes[nrclasses][10] = { "beach","city","desert","forest","landscape","snow" };
	Mat X(672, 3 * m, CV_32FC1);
	Mat Y(672, 1, CV_8UC1);
	int fileNr, rowX = 0;
	char fname[256];
	for (int c = 0; c < nrclasses; c++) {
		fileNr = 0;
		while (1) {
			sprintf(fname, "train/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) {
				break;
			}
			int *hist = computeHistogram(img, m);
			for (int d = 0; d < 3 * m; d++) {
				X.at<float>(rowX, d) = hist[d];
			}
			Y.at<uchar>(rowX) = c;
			rowX++;
		}
	}
	/*
	Mat Xt(85, 3 * m, CV_32FC1);
	Mat Yt(85, 1, CV_8UC1);
	int fileNrT, rowXT = 0;
	char fnameT[256];
	for (int c = 0; c < nrclasses; c++) {
		fileNrT = 0;
		while (1) {
			sprintf(fnameT, "test/%s/%06d.jpeg", classes[c], fileNrT++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) {
				break;
			}
			int *hist = computeHistogram(img, m);
			for (int d = 0; d < 3 * m; d++) {
				Xt.at<float>(rowXT, d) = hist[d];
			}
			Yt.at<uchar>(rowXT) = c;
			rowXT++;
		}
	}

	Mat C = Mat::zeros(nrclasses, nrclasses, CV_8UC1);
	int *testHist = (int*)malloc(3 * m * sizeof(int));

	for (int z = 0; z < 85; z++) {
		float distances[672];
		for (int j = 0; j < m * 3; j++) {
			testHist[j] = Xt.at<float>(z, j);
		}
		for (int i = 0; i < 672; i++) {
			distances[i] = 0.0;
			for (int j = 0; j < m * 3; j++) {
				distances[i] += (testHist[j] - X.at<float>(i, j)) * (testHist[j] - X.at<float>(i, j));
			}
			distances[i] = sqrt(distances[i]);
		}
		float unsortedDistances[672];
		for (int i = 0; i < 672; i++) {
			unsortedDistances[i] = distances[i];
		}

		std::qsort(distances, 672, sizeof(float), compare);

		int votes[nrclasses];
		for (int i = 0; i < nrclasses; i++) {
			votes[i] = 0;
		}
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < 672; j++) {
				if (distances[i] == unsortedDistances[j]) {
					votes[Y.at<uchar>(j)]++;
				}
			}
		}
		int max = -1, index = -1;
		for (int i = 0; i < nrclasses; i++) {
			if (votes[i] > max) {
				max = votes[i];
				index = i;
			}
		}
		for (int c = 0; c < nrclasses; c++) {
			if (index == c) {
				std::cout << classes[c];
			}
		}
		C.at<float>(index, (int)Yt.at<uchar>(index))++;
	}

	float acc = 0.0;
	float sumd = 0.0;
	for (int i = 0; i < nrclasses; i++) {
		sumd += C.at<float>(i, i);
	}
	float sum = 0.0;
	for (int i = 0; i < nrclasses; i++) {
		for (int j = 0; j < nrclasses; j++) {
			sum += C.at<float>(i, j);
		}
	}
	acc = sumd / sum * 100;
	std::cout << "Accuracy : " << acc << "%" << std::endl;
	*/
	float distances[672];
	Mat testImg = imread("test/beach/000002.jpeg", CV_LOAD_IMAGE_COLOR);
	int *testHist = computeHistogram(testImg, m);
	for (int i = 0; i < 672; i++) {
		distances[i] = 0.0;
		for (int j = 0; j < m * 3; j++) {
			distances[i] += (testHist[j] - X.at<float>(i, j)) * (testHist[j] - X.at<float>(i, j));
		}
		distances[i] = sqrt(distances[i]);
	}
	float unsortedDistances[672];
	for (int i = 0; i < 672; i++) {
		unsortedDistances[i] = distances[i];
	}

	std::qsort(distances, 672, sizeof(float), compare);

	int votes[nrclasses];
	for (int i = 0; i < nrclasses; i++) {
		votes[i] = 0;
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < 672; j++) {
			if (distances[i] == unsortedDistances[j]) {
				votes[Y.at<uchar>(j)]++;
			}
		}
	}
	int max = -1, index = -1;
	for (int i = 0; i < nrclasses; i++) {
		if (votes[i] > max) {
			max = votes[i];
			index = i;
		}
	}
	for (int c = 0; c < nrclasses; c++) {
		if (index == c) {
			std::cout << classes[c];
		}
	}
	
	getchar();
	getchar();
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

void bayes() {

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

	std::fstream wfile;
	wfile.open("bayes/likelihood.txt");
	for (int i = 0; i < nrclasses; i++) {
		for (int j = 0; j < d; j++) {
			wfile << likelihood.at<double>(i, j) << "  ";
		}
		wfile << std::endl;
	}

	double p[nrclasses];
	int count = 0;
	Mat testImg = imread("bayes/test/8/000348.png", CV_LOAD_IMAGE_GRAYSCALE);
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

	getchar();
	getchar();
}

void perceptron() {

	Mat img = imread("perceptron/test05.bmp", CV_LOAD_IMAGE_COLOR);
	int D[] = { -1, 1 };
	int pointsNo = 0;
	Point2d *points = (Point2d*)malloc(10000*sizeof(Point2d));
	int *classes = (int*)malloc(10000*sizeof(int));
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j)[0] == 255 && img.at<Vec3b>(i, j)[1] == 0 && img.at<Vec3b>(i, j)[2] == 0) {
				points[pointsNo] = Point2d(j, i);
				classes[pointsNo] = D[1];
				pointsNo++;
			}
			if (img.at<Vec3b>(i, j)[2] == 255 && img.at<Vec3b>(i, j)[0] == 0 && img.at<Vec3b>(i, j)[1] == 0) {
				points[pointsNo] = Point2d(j, i);
				classes[pointsNo] = D[0];
				pointsNo++;
			}
		}
	}

	Mat X(pointsNo, 3, CV_64FC1);
	Mat Y(pointsNo, 1, CV_32SC1);

	for (int i = 0; i < pointsNo; i++) {
		X.at<double>(i, 0) = 1.0;
		X.at<double>(i, 1) = points[i].x;
		X.at<double>(i, 2) = points[i].y;
		Y.at<int>(i) = classes[i];
	}

	double w[3] = { 0.1, 0.1, 0.1 };
	double eta = 0.0001;
	double eLimit = 0.00001;
	double maxIter = 100000;
	double E, L;
	double z = 0.0;


	for (int iter = 0; iter < maxIter; iter++) {
		E = 0.0;
		L = 0.0;
		double grad[3] = { 0.0,0.0,0.0 };
		for (int i = 0; i < pointsNo; i++) {
				z = 0.0;
				for (int j = 0; j < 3; j++) {
					z += w[j] * X.at<double>(i, j);
				}
				if (z * Y.at<int>(i) <= 0.0) {
					for (int k = 0; k < 3; k++) {
						grad[k] -= Y.at<int>(i) * X.at<double>(i, k);
					}
					E += 1.0;
					L -= z * Y.at<int>(i);
				}
			}
			E = E / (double)pointsNo;
			L = L / (double)pointsNo;
		//	for (int j = 0; j < 3; j++) {
		//		grad[j] = grad[j] / (double)pointsNo;
		//	}
			if (E < eLimit) {
				break;
			}
			for (int j = 0; j < 3; j++) {
				w[j] -= eta * grad[j];
			}
	}

	line(img, Point(0, -w[0] / w[2]), Point((img.cols - 1), (-w[0] - w[1] * (img.cols - 1)) / w[2]) , Scalar(0, 255, 0), 1);

	std::cout << pointsNo << std::endl;
	std::cout << w[0] << " " << w[1] << " " << w[2];
	imshow("img", img);
	waitKey(0);

}

struct weakLearner {
	int feature_i;
	int threshold;
	int class_label;
	double error;
	int classify(Mat X) {
		if (X.at<double>(feature_i) < threshold) {
			return class_label;
		}
		else {
			return -class_label;
		}
	}
};

struct classifier {
	int T;
	float alphas[1000];
	weakLearner hs[1000];
	int classify(Mat X) {
		int s = 0;
		for (int i = 0; i < T; i++) {
			s += hs[i].classify(X.row(i));
		}
		if (s > 0) {
			return 1;
		}
		else {
			return -1;
		}
	}
};

weakLearner findWeakLearner(Mat X, Mat Y, Mat w, int size) {
	weakLearner best_h;
	double best_err = FLT_MAX;
	int class_labels[2] = { -1, 1 };
	Mat z(X.rows, 1, CV_64FC1);
	for (int j = 1; j < X.cols; j++) {
		for (int threshold = 0; threshold < size; threshold++) {
			for (int c = 0; c < 2; c++) {
				double e = 0.0;
				for (int i = 1; i < X.rows; i++) {
					if (X.at<double>(i, j) < threshold) {
						z.at<double>(i) = class_labels[c];
					}
					else {
						z.at<double>(i) = -class_labels[c];
					}
					if (z.at<double>(i) * Y.at<double>(i) < 0) {
						e += w.at<double>(i);
					}
				}
				if (e < best_err) {
					best_err = e;
					best_h.feature_i = j;
					best_h.threshold = threshold;
					best_h.class_label = class_labels[c];
					best_h.error = e;
				}
			}
		}
	}
	return best_h;
}

void drawBoundary(Mat img, classifier clf) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i,j)[0] == 255 && img.at<Vec3b>(i,j)[1] == 255 && img.at<Vec3b>(i,j)[2] == 255) {
				Mat X(1, 2, CV_64FC1);
				X.at<double>(0, 0) = (double)i;
				X.at<double>(0, 1) = (double)j;
				if (clf.classify(X) == 1) {
					img.at<Vec3b>(i, j)[0] = 0;
					img.at<Vec3b>(i, j)[1] = 255;
					img.at<Vec3b>(i, j)[2] = 255;
				}
				else {
					img.at<Vec3b>(i, j)[0] = 255;
					img.at<Vec3b>(i, j)[1] = 255;
					img.at<Vec3b>(i, j)[2] = 0;
				}
			}
		}
	}
}

void adaboost(int T) {
	Mat img = imread("adaboost/points0.bmp", CV_LOAD_IMAGE_COLOR);

	int D[] = { -1, 1 };
	int pointsNo = 0;
	Point2d *points = (Point2d*)malloc(10000 * sizeof(Point2d));
	int *classes = (int*)malloc(10000 * sizeof(int));
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j)[0] == 255 && img.at<Vec3b>(i, j)[1] == 0 && img.at<Vec3b>(i, j)[2] == 0) {
				points[pointsNo] = Point2d(j, i);
				classes[pointsNo] = D[1];
				pointsNo++;
			}
			if (img.at<Vec3b>(i, j)[2] == 255 && img.at<Vec3b>(i, j)[0] == 0 && img.at<Vec3b>(i, j)[1] == 0) {
				points[pointsNo] = Point2d(j, i);
				classes[pointsNo] = D[0];
				pointsNo++;
			}
		}
	}

	Mat X(pointsNo, 2, CV_64FC1);
	Mat Y(pointsNo, 1, CV_64FC1);
	Mat w(pointsNo, 1, CV_64FC1);

	for (int i = 0; i < pointsNo; i++) {
		X.at<double>(i, 0) = points[i].x;
		X.at<double>(i, 1) = points[i].y;
		Y.at<double>(i) = (double)classes[i];
		w.at<double>(i) = 1.0 / pointsNo;
	}

	classifier adaBoost;
	double *alfa = (double*)malloc(T * sizeof(double));
	for (int t = 1; t < T; t++) {
		weakLearner h = findWeakLearner(X, Y, w, img.rows);
		alfa[t] = 0.5 * (log((1 - h.error) / h.error));
		double s = 0.0;
		for (int i = 0; i < pointsNo; i++) {
			w.at<double>(i) = w.at<double>(i) * exp(-alfa[t] * Y.at<double>(i) * h.classify(X.row(i)));
			s += w.at<double>(i);
		}
		for (int i = 0; i < pointsNo; i++) {
			w.at<double>(i) = w.at<double>(i) / s;
		}
		adaBoost.alphas[t] = alfa[t];
		adaBoost.hs[t] = h;
		adaBoost.T = T;
	}

	drawBoundary(img, adaBoost);
	imshow("img", img);
	waitKey(0);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 11 - Least Mean Squares\n");
		printf(" 12 - RANSAC\n");
		printf(" 13 - Hough transform\n");
		printf(" 14 - Distance transform and pattern matching\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 11:
				leastMean();
				break;
			case 12:
				ransac();
				break;
			case 13:
				houghTransform();
				break;
			case 14:
				distanceTransform();
				break;
			case 15:
				statistics();
				break;
			case 16:
				KmeansClustering(3);
				break;
			case 17:
				pca();
				break;
			case 18:
				kNN();
				break;
			case 19:
				bayes();
				break;
			case 20:
				perceptron();
				break;
			case 21:
				adaboost(13);
				break;
		}
	}
	while (op!=0);
	return 0;
}