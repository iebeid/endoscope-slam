#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>

#include "optical_flow.hpp"
#include "fps.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;

int calibrate(int number_of_calibration_images)
{
	int numBoards = number_of_calibration_images;
	int numCornersHor = 9;
	int numCornersVer = 6;
	int numSquares = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);
	vector<vector<Point3f>> object_points;
	vector<vector<Point2f>> image_points;
	vector<Point2f> corners;
	int successes = 0;
	vector<Point3f> obj;

	for (int j = 0; j < numSquares; j++)
		obj.push_back(Point3f((float)j / (float)numCornersHor, (float)(j%numCornersHor), 0.0f));

	VideoCapture cap(1);
	Mat image;
	cap >> image;
	while (successes < numBoards){
		cap >> image;
		Mat gray_image;
		cv::cvtColor(image, gray_image, CV_BGR2GRAY);
		bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		if (found)
		{
			cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray_image, board_sz, corners, found);
		}
		cv::imshow("Endoscope", gray_image);
		int key = waitKey(60);
		if (key == 27)
			return 0;

		if (key == 13 && found)
		{
			image_points.push_back(corners);
			object_points.push_back(obj);
			std::cout << "Calibration Image Stored" << std::endl;
			successes++;
			if (successes >= numBoards)
				break;
		}
	}

	destroyAllWindows();
	//Calibrate
	Mat K;
	Mat D;
	calibrateCamera(object_points, image_points, image.size(), K, D, noArray(), noArray(), 0, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
	//std::cout << K << std::endl;
	//std::cout << D << std::endl;

	ofstream myfile("camera.txt");
	if (myfile.is_open()){
		myfile << K.at<double>(0, 0) << endl;
		myfile << K.at<double>(0, 1) << endl;
		myfile << K.at<double>(0, 2) << endl;
		myfile << K.at<double>(1, 0) << endl;
		myfile << K.at<double>(1, 1) << endl;
		myfile << K.at<double>(1, 2) << endl;
		myfile << K.at<double>(2, 0) << endl;
		myfile << K.at<double>(2, 1) << endl;
		myfile << K.at<double>(2, 2) << endl;
		myfile << D.at<double>(0, 0) << endl;
		myfile << D.at<double>(0, 1) << endl;
		myfile << D.at<double>(0, 2) << endl;
		myfile << D.at<double>(0, 3) << endl;
		myfile.close();
	}
	else{
		std::cout << "Unable to open file" << endl;
	}
	
	cap.release();
	return 0;
}

int load_calibration(double * kvalues, double * dvalues, const char * filename){
	string line;
	ifstream myfile(filename);
	if (myfile.is_open())
	{
		//*kvalues = new double[9];
		//*dvalues = new double[4];
		int i = 0;
		int k = 0;
		int d = 0;
		while (getline(myfile, line))
		{
			if (i < 9){
				kvalues[k] = stod(line);
				k++;
			}
			else{
				dvalues[d] = stod(line);
				d++;
			}
			i++;
		}

		//K->at<double>(0, 0) = kvalues[0];
		//K->at<double>(0, 1) = kvalues[1];
		//K->at<double>(0, 2) = kvalues[2];
		//K->at<double>(1, 0) = kvalues[3];
		//K->at<double>(1, 1) = kvalues[4];
		//K->at<double>(1, 2) = kvalues[5];
		//K->at<double>(2, 0) = kvalues[6];
		//K->at<double>(2, 1) = kvalues[7];
		//K->at<double>(2, 2) = kvalues[8];
		//D->at<double>(0, 0) = dvalues[0];
		//D->at<double>(0, 1) = dvalues[1];
		//D->at<double>(0, 2) = dvalues[2];
		//D->at<double>(0, 3) = dvalues[3];

		//cout << *K << endl;
		//cout << *D << endl;
		//cout << "---------------------------" << endl;
		myfile.close();
		return 0;
	}
	else{
		cout << "Unable to open file" << endl;
		return -1;
	}
}

int run_with_cuda(){ 
	VideoCapture cap(1);
	Mat image;
	cap >> image;
	while (1){
		cap >> image;
		Mat gray_image;
		cv::cvtColor(image, gray_image, CV_BGR2GRAY);
		
		cv::imshow("Endoscope", gray_image);

		//cv::Mat mask(image.size(), CV_8UC1, cv::Scalar::all(1));
		//mask(cv::Range(0, image.rows / 2), cv::Range(0, image.cols / 2)).setTo(cv::Scalar::all(0));


		

		//Ptr<cuda::ORB> d_orb = cuda::ORB::create();
		//std::vector<cv::KeyPoint> keypoints;
		//cv::cuda::GpuMat d_descriptors;
		//d_orb->detectAndCompute(gray_image, noArray(), keypoints, d_descriptors);

		//int nfeatures = 500;
		//float scaleFactor = 1.2f;
		//int nlevels = 8;
		//int edgeThreshold = 31;
		//int firstLevel = 0;
		//int WTA_K = 2;
		//int scoreType = ORB::HARRIS_SCORE;
		//int patchSize = 31;
		//int fastThreshold = 20;

		//cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, 20, false);

		//std::vector<cv::KeyPoint> keypoints;
		//cv::cuda::GpuMat descriptors;
		//orb->detectAndComputeAsync(loadMat(image), loadMat(mask), keypoints, descriptors);
		
		//cv::Ptr<cv::ORB> orb_gold = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);

		//std::vector<cv::KeyPoint> keypoints_gold;
		//cv::Mat descriptors_gold;
		//orb_gold->detectAndCompute(image, mask, keypoints_gold, descriptors_gold);

		//cv::BFMatcher matcher(cv::NORM_HAMMING);
		//std::vector<cv::DMatch> matches;
		//matcher.match(descriptors_gold, cv::Mat(descriptors), matches);

		//int matchedCount = getMatchedPointsCount(keypoints_gold, keypoints, matches);
		//double matchedRatio = static_cast<double>(matchedCount) / keypoints.size();

		int key = waitKey(60);
		if (key == 27)
			return 0;
	}
	cap.release();
	return 0; 
}


int calculate_optical_flow(Mat frame0, Mat frame1){


	GpuMat d_frame0(frame0);
	GpuMat d_frame1(frame1);

	GpuMat d_flow(frame0.size(), CV_32FC2);

	//Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
	Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(3, 3));
	//Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
	//Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();

	//{
	//	GpuMat d_frame0f;
	//	GpuMat d_frame1f;

	//	d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	//	d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

	//	const int64 start = getTickCount();

	//	brox->calc(d_frame0f, d_frame1f, d_flow);

	//	const double timeSec = (getTickCount() - start) / getTickFrequency();
	//	cout << "Brox : " << timeSec << " sec" << endl;

		//showFlow("Brox", d_flow);
	//}

	{
		const int64 start = getTickCount();

		lk->calc(d_frame0, d_frame1, d_flow);

		const double timeSec = (getTickCount() - start) / getTickFrequency();
		cout << "LK : " << timeSec << " sec" << endl;

		showFlow("LK", d_flow);
	}

	//{
	//	const int64 start = getTickCount();

	//	farn->calc(d_frame0, d_frame1, d_flow);

	//	const double timeSec = (getTickCount() - start) / getTickFrequency();
	//	cout << "Farn : " << timeSec << " sec" << endl;

	//	showFlow("Farn", d_flow);
	//}

	//{
	//	const int64 start = getTickCount();

	//	tvl1->calc(d_frame0, d_frame1, d_flow);

	//	const double timeSec = (getTickCount() - start) / getTickFrequency();
	//	cout << "TVL1 : " << timeSec << " sec" << endl;

	//	showFlow("TVL1", d_flow);
	//}

	//imshow("Frame 0", frame0);
	//imshow("Frame 1", frame1);
	//waitKey();

	return 0;
}

struct orb_result{
	cuda::GpuMat descriptors;
	vector<KeyPoint> keypoints;
};

//http://study.marearts.com/2014/07/opencv-study-orb-gpu-feature-extraction.html
orb_result extract_orb(Mat image){
	GpuMat d_frame(image);
	int nfeatures = 500;
	float scaleFactor = 1.2f;
	int nlevels = 8;
	int edgeThreshold = 31;
	int firstLevel = 0;
	int WTA_K = 2;
	int scoreType = cuda::ORB::HARRIS_SCORE;
	int patchSize = 31;
	int fastThreshold = 20;
	Ptr<cuda::ORB> orb_object = cuda::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, 20, false);
	vector<KeyPoint> keypoints;
	cuda::GpuMat descriptors;
	Mat mask(image.size(), CV_8UC1, Scalar::all(1));
	mask(Range(0, image.rows / 2), Range(0, image.cols / 2)).setTo(Scalar::all(0));
	cuda::GpuMat d_mask(mask);
	orb_object->detectAndCompute(d_frame, d_mask, keypoints, descriptors);
	orb_result result;
	result.descriptors = descriptors;
	result.keypoints = keypoints;
	return result;
}
//http://www.morethantechnical.com/2012/02/07/structure-from-motion-and-3d-reconstruction-on-the-easy-in-opencv-2-3-w-code/
//http://stackoverflow.com/questions/20467403/opencv-fundamental-matrix-and-moving-camera
//http://www.morethantechnical.com/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/
//http://answers.opencv.org/question/29063/how-to-save-current-frame-and-previous-frame/   saving the current and the previous frame
int run_stream(){ 
	VideoCapture cap(1);
	Mat image, prev;
	cap >> image;
	cv::cvtColor(image, image, CV_BGR2GRAY);
	image.copyTo(prev);
	fps *counter = new fps();
	int frame_counter = 0;
	double kvalues[9];
	double dvalues[4];
	int result = load_calibration(kvalues, dvalues, "camera.txt");
	Mat K = Mat(Size(3, 3), CV_64F, &kvalues);
	Mat D = Mat(Size(1, 4), CV_64F, &dvalues);
	cout << K << endl;
	cout << D << endl;
	//vector<Mat> projections;
	//int projection_counter = 0;
	while (1){
		counter->start_fps_counter();
		cap >> image;
		cv::cvtColor(image, image, CV_BGR2GRAY);
		cv::imshow("Endoscope Current Frame", image);
		cv::imshow("Endoscope Previous Frame", prev);
		
		if (frame_counter>5){
			//calculate_optical_flow(prev, image);

			//ORB Feature extraction and matching
			orb_result orb_descriptors_current = extract_orb(image);
			orb_result orb_descriptors_previous = extract_orb(prev);
			Ptr<cuda::DescriptorMatcher> matcher_object = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
			vector< vector< DMatch> > matches;
			matcher_object->knnMatch(orb_descriptors_previous.descriptors, orb_descriptors_current.descriptors, matches, 2);
			std::vector< DMatch > good_matches;
			for (int k = 0; k < min(orb_descriptors_previous.descriptors.rows - 1, (int)matches.size()); k++)
			{
				if ((matches[k][0].distance < 0.7*(matches[k][1].distance)) && ((int)matches[k].size() <= 2 && (int)matches[k].size()>0))
				{
					good_matches.push_back(matches[k][0]);
				}
			}
			Mat img_matches;
			cv::drawMatches(prev, orb_descriptors_previous.keypoints, image, orb_descriptors_current.keypoints, good_matches, img_matches);
			cv::imshow("Matching keypoints", img_matches);
			
			//int point_count = 100;
			vector<Point2d> points1;
			vector<Point2d> points2;
			//vector<Point3d> triangulated_points;
			//cout << good_matches.size() << endl;
			for (unsigned int i = 0; i<good_matches.size(); i++)
			{
				points1.push_back(orb_descriptors_previous.keypoints[good_matches[i].trainIdx].pt);
			    points2.push_back(orb_descriptors_current.keypoints[good_matches[i].queryIdx].pt);
			}

			//for (int i = 0; i < point_count; i++)
			//{
			//	points1[i] = orb_descriptors_previous.keypoints.at(i).pt;
			//	points2[i] = orb_descriptors_current.keypoints.at(i).pt;
			//	
			//}

			if (good_matches.size() > 0){
				Mat mask;
				Mat fundamental_matrix = Mat(Size(3, 3), CV_64F, float(0));
				fundamental_matrix = cv::findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99, mask);
				if (!fundamental_matrix.empty() && fundamental_matrix.cols == K.rows && fundamental_matrix.rows == K.cols){
					Mat E = K.t() * fundamental_matrix * K;
					SVD svd(E);
					Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);
					Matx33d Winv(0, 1, 0, -1, 0, 0, 0, 0, 1);
					Mat R = svd.u * Mat(W) * svd.vt;
					Mat t = svd.u.col(2);

					double P00 = R.at<double>(Point(0, 0));
					double P01 = R.at<double>(Point(0, 1));
					double P02 = R.at<double>(Point(0, 2));
					double P03 = t.at<double>(0);

					double P04 = R.at<double>(Point(1, 0));
					double P05 = R.at<double>(Point(1, 1));
					double P06 = R.at<double>(Point(1, 2));
					double P07 = t.at<double>(1);

					double P08 = R.at<double>(Point(2, 0));
					double P09 = R.at<double>(Point(2, 1));
					double P10 = R.at<double>(Point(2, 2));
					double P11 = t.at<double>(2);

					double data [12] = { P00, P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11 };

					Mat RT = Mat(3, 4, CV_64FC1, &data);

					if (!RT.empty() && RT.cols == K.rows && RT.rows == K.cols){

						

	/*					try
						{
							Mat P = K * RT;
						}
						catch (cv::Exception& e)
						{
							const char* err_msg = e.what();
							std::cout << "exception caught: " << err_msg << std::endl;
						}*/
					}

					

					//projections.push_back(P);
					
					//if (frame_counter>frame_counter+1){

						int N = (int)good_matches.size();
						
						cv::Mat pnts3D;
						//cv::Mat cam0pnts(1, N, CV_64FC2);
						//cv::Mat cam1pnts(1, N, CV_64FC2);
					//cv::Mat pnts3D(1, N, CV_64FC4);
					//cv::Mat cam0pnts(1, N, CV_64FC2);
					//cv::Mat cam1pnts(1, N, CV_64FC2);
					//cv::triangulatePoints(P, P, points1, points2, pnts3D);

					//for (int y = 0; y < pnts3D.rows; y++)
					//{
						//for (int x = 0; x < pnts3D.cols; x++)
						//{
							//Vec4d point = pnts3D.at<Vec4d>(Point(x, y));
							//cout << "X: " << point.val[0] << "  Y: " << point.val[1] << "  Z: " << point.val[2] << endl;
							//cout << pnts3D.at<Vec4f>(Point(x, y))[0] << endl;
							//cout << pnts3D.at<Vec4f>(Point(x, y))[1] << endl;
							//cout << pnts3D.at<Vec4f>(Point(x, y))[2] << endl;
							//cout << pnts3D.at<Vec4f>(Point(x, y))[3] << endl;
						//}
					//}


				}
			}

			

			//Mat E = K.t() * fundamental_matrix * K;

			////-- Stereo Matching
			//Mat imgDisparity16S = Mat(prev.rows, prev.cols, CV_16S);
			//Mat imgDisparity8U = Mat(prev.rows, prev.cols, CV_8UC1);
			//if (image.empty() || image.empty())
			//{
			//	std::cout << " --(!) Error reading images " << std::endl; return -1;
			//}
			//int ndisparities = 16 * 5;
			//int SADWindowSize = 21;
			//Ptr<cv::StereoBM> sbm = cv::StereoBM::create(ndisparities, SADWindowSize);
			//sbm->compute(prev, image, imgDisparity16S);
			//double minVal; double maxVal;
			//minMaxLoc(imgDisparity16S, &minVal, &maxVal);
			//imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
			//imshow("Disparity Image", imgDisparity8U);

		}
		image.copyTo(prev);
		int key = waitKey(60);
		if (key == 27){
			return 0;
		}
		else if (key == 32){
			cout << "Snapshot Stored" << endl;
			imwrite("current_snapshot.jpg", image);
		}
		frame_counter++;
		counter->end_fps_counter();
		counter->print_fps();
	}
	cap.release();
	return 0;
}

int kmeans(Mat img)
{
	const int MAX_CLUSTERS = 5;
	Scalar colorTab[] =
	{
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(255, 100, 100),
		Scalar(255, 0, 255),
		Scalar(0, 255, 255)
	};

	//Mat img(500, 500, CV_8UC3);
	RNG rng(12345);

	for (;;)
	{
		int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
		int i, sampleCount = rng.uniform(1, 1001);
		Mat points(sampleCount, 1, CV_32FC2), labels;

		clusterCount = MIN(clusterCount, sampleCount);
		Mat centers;

		/* generate random sample from multigaussian distribution */
		for (k = 0; k < clusterCount; k++)
		{
			Point center;
			center.x = rng.uniform(0, img.cols);
			center.y = rng.uniform(0, img.rows);
			Mat pointChunk = points.rowRange(k*sampleCount / clusterCount,
				k == clusterCount - 1 ? sampleCount :
				(k + 1)*sampleCount / clusterCount);
			rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
		}

		randShuffle(points, 1, &rng);

		kmeans(points, clusterCount, labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
			3, KMEANS_PP_CENTERS, centers);

		img = Scalar::all(0);

		for (i = 0; i < sampleCount; i++)
		{
			int clusterIdx = labels.at<int>(i);
			Point ipt = points.at<Point2f>(i);
			circle(img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
		}
		cv::imshow("clusters", img);
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}

	return 0;
}


int main(int argc, const char * argv[])
{
	cout << "OpenCV version : " << CV_VERSION << endl;
	cout << "--------------------------" << endl;
	int result=0;
	//result = calibrate(10);

	//double kvalues [9];
	//double dvalues [4];
	//result = load_calibration(kvalues, dvalues, "camera.txt");
	//Mat K = Mat(Size(3, 3), CV_64F, &kvalues);
	//Mat D = Mat(Size(1, 4), CV_64F, &dvalues);
	//cout << K << endl;
	//cout << D << endl;
	//getchar();
	result = run_stream();
	//result = run_with_cuda();
	//Mat image;
	//image = imread("current_snapshot.jpg", 1);
	//result = kmeans(image);
	return result;
}