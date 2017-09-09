#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
/*****************************************************************************************************************************/
using namespace cv;
using namespace cv::ml;
/*****************************************************************************************************************************/
const std::string IMAGETYPE = ".jpg";
const unsigned int NUMBEROFCLASSES = 1;
const unsigned int NUMBEROFIMAGESPERCLASS = 1;
/*****************************************************************************************************************************/
Mat resizeImageTo1xN(Mat image);
Mat populateTrainingMat(const unsigned int numberOfImages, const unsigned int numberOfClasses);
std::vector<int> populateLabels(const unsigned int numberOfImages, const unsigned int numberOfClasses);
/*****************************************************************************************************************************/


int main(int, char**)
{
	Mat trainingMatrix = populateTrainingMat(1,1);
	//Mat test = imread("C:\\Users\\Casa\\Desktop\\1.jpg",IMREAD_GRAYSCALE);
	//Size size(test.cols*test.rows,1);
	//std::vector<int> labels;
	//resize(test,test,size);
	//labels=populateLabels(10,3);
	//for (int j=0 ; j<30;j++){

	//				std::cout<<labels[j]<<" ";
	//			}



	/*
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Set up training data
    int labels[4] = { 1, -1, -1, -1 };
    Mat labelsMat(4, 1, CV_32SC1, labels);

    float trainingData[4][2] = { { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 } };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    // Set up SVM's parameters
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    // Train the SVM with given parameters
    Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
    svm->train(td);

    // Or train the SVM with optimal parameters
    //svm->trainAuto(td);

    Vec3b green(0, 255, 0), blue(255, 0, 0);
    // Show the decision regions given by the SVM
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1, 2) << j, i);
            float response = svm->predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(i, j) = green;
            else if (response == -1)
                image.at<Vec3b>(i, j) = blue;
        }

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
    circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

    // Show support vectors
    thickness = 2;
    lineType = 8;
    Mat sv = svm->getSupportVectors();

    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("result.png", image);        // save the image
*/
   // imshow("SVM Simple Example", trainingMatrix); // show it to the user
    std::cout<<(int)trainingMatrix.at<uchar>(0,trainingMatrix.cols);
    waitKey(0);

}

/*****************************************************************************************************************************/
Mat resizeImageTo1xN(Mat image){
	Size size(image.cols*image.rows,1);
	resize(image,image,size);
	return image;
}

/*****************************************************************************************************************************/
Mat populateTrainingMat(const unsigned int numberOfImages, const unsigned int numberOfClasses){
	// TO DO
	// CORRECT THE IMREAD NAME PARAMETER
	char *sampleImageName = "0"+IMAGETYPE.c_str();
	Mat sampleImage = imread(sampleImageName,IMREAD_GRAYSCALE);
	Mat trainingMatrix = Mat::zeros(numberOfClasses*numberOfImages,sampleImage.cols*sampleImage.rows+1,CV_8UC1);
	for (unsigned int i=0; i<numberOfClasses;i++){
		for (unsigned int j=0 ; j<numberOfImages;j++){
			const unsigned int imageNumber = j+(numberOfImages*i);
			char *imageName = imageNumber + IMAGETYPE.c_str();
 			Mat input=imread(imageName ,IMREAD_GRAYSCALE);
			Mat resizedInput = resizeImageTo1xN(input);
			resizedInput.copyTo(trainingMatrix(Rect(0,j+(numberOfImages*i),resizedInput.cols,resizedInput.rows)));
			trainingMatrix.at<uchar>(j,trainingMatrix.cols+1)=i;
		}
	}
	return trainingMatrix;
}
/*****************************************************************************************************************************/
std::vector<int> populateLabels(const unsigned int numberOfImages, const unsigned int numberOfClasses){
	int size = numberOfImages *numberOfClasses;
	std::vector<int> labels(size);
	for (unsigned int i=0; i<numberOfClasses;i++){
			for (unsigned int j=0 ; j<numberOfImages;j++){
				labels.at(j+(numberOfImages*i))=i;
			}
	}
	return labels;
}

/*****************************************************************************************************************************/
