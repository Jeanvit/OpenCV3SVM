#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

/*****************************************************************************************************************************/
//Namespaces
using namespace cv;
using namespace cv::ml;
using std::cout;
using std::endl;

/*****************************************************************************************************************************/
//Global
const std::string IMAGETYPE = ".jpg";              //Image type for all images inside the program
const unsigned int NUMBEROFCLASSES = 2;		   //Number of classes used in the Training
const unsigned int NUMBEROFIMAGESPERCLASS =10;     //Number of Images in each class
const unsigned int IMAGERESOLUTION = 256;          //Y and X resolution for all images

/*****************************************************************************************************************************/
//Headers
Mat resizeTo1xN(Mat image);
Mat populateTrainingMat(const unsigned int numberOfImages, const unsigned int numberOfClasses);
Mat populateLabels(const unsigned int numberOfImages, const unsigned int numberOfClasses);
void printSupportVectors(const Ptr<SVM>& svm);
Mat edgeDetection(Mat image);
/*****************************************************************************************************************************/
//Main
auto main(int argc, char *argv[]) -> int {
	Mat testImage;
	if (argc>1){
		std::string image = argv[1];
		testImage = imread(image.c_str(),IMREAD_GRAYSCALE);
		if (!testImage.data){
			cout<<"Error opening the image!";
			return (0);
		}
		cout<<"Parameters: "<<"extension: "<<IMAGETYPE<<" Nclasses: "<<NUMBEROFCLASSES<<" Nimages: "<<NUMBEROFIMAGESPERCLASS<<" Resolution: "<<IMAGERESOLUTION<<endl;
	}
	else {
		cout<<"Please, specify the test image!";
		return(0);
	}

	Mat sampleImage = resizeTo1xN(edgeDetection(testImage));
	sampleImage.convertTo(sampleImage,CV_32F);
	Mat trainingMatrix = populateTrainingMat(NUMBEROFIMAGESPERCLASS,NUMBEROFCLASSES);
	Mat labels=populateLabels(NUMBEROFIMAGESPERCLASS,NUMBEROFCLASSES);
	cout<<"Train data"<<endl<<"rows: "<<trainingMatrix.rows<<" cols: "<<trainingMatrix.cols<<endl<<endl;
	cout<<"Labels"<<endl<<"rows: "<<labels.rows<<" cols: "<<labels.cols<<endl<<endl;
	cout<<"Prediction Image"<<endl<<"rows: "<<sampleImage.rows<<" cols: "<<sampleImage.cols<<endl<<endl;

    // Set up SVM's parameters
    cout<<"Setting up SVM's parameters.";
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC );
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
    cout<<"..Done!"<<endl;

    // Train the SVM with given parameters
    Ptr<TrainData> td = TrainData::create(trainingMatrix, ROW_SAMPLE, labels);
    cout<<"Training SVM.";
    svm->train(td);
    cout<<"..Done!"<<endl<<endl;

    // Train the SVM with optimal parameters
    //svm->trainAuto(td);

    float response = svm->predict(sampleImage);
    cout<<"The test image belongs to class: "<<response<<endl;

    printSupportVectors(svm);

    waitKey(0);
}

/******************************************************************************************************************************/
Mat resizeTo1xN(Mat image){
	/* This function resize the given image into a 1xN Mat
	 * @param Mat image - The image to be resized
	 */
	Size size(IMAGERESOLUTION,IMAGERESOLUTION);
	resize(image,image,size);
	Size size1xN(image.cols*image.rows,1);
	resize(image,image,size1xN);
	return image;

}
/*****************************************************************************************************************************/
Mat populateTrainingMat(const unsigned int numberOfImages, const unsigned int numberOfClasses){
	/*Build the trainig Matrix for the the SVM. Each line will be an image.
	 * @param  const unsigned int numberOfImages
	 * @param const unsigned int numberOfClasses
	 *
	 */
	Mat trainingMatrix = Mat::zeros(numberOfClasses*numberOfImages,IMAGERESOLUTION*IMAGERESOLUTION,CV_32F);
	for (unsigned int i=0; i<numberOfClasses;i++){
		for (unsigned int j=0 ; j<numberOfImages;j++){
			const unsigned int imageNumber = j+(numberOfImages*i);
			std::ostringstream ss;
			ss<<imageNumber;                                                       // Workaround for std::to_string MinGW bug
			cv::String imageName =  ss.str() + IMAGETYPE;
 			Mat input=imread(imageName ,IMREAD_GRAYSCALE);
 			input=edgeDetection(input);
 			Mat resizedInput=resizeTo1xN(input);
			resizedInput.copyTo(trainingMatrix(Rect(0,j+(numberOfImages*i),resizedInput.cols,resizedInput.rows)));
		}
	}
	return trainingMatrix;
}

/*****************************************************************************************************************************/
Mat edgeDetection(Mat image){
	/*Uses the Sobel–Feldman operator to extract the image edges
			 * @param  Mat image - The image to be processed
			 *
			 */
	 int scale = 1;
	 int delta = 0;
	 int ddepth = CV_32F;
	 Mat gradX, gradY;
	 Mat absGradX, absGradY;

	 // Gradient X
	 Sobel( image, gradX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	 convertScaleAbs( gradX, absGradX );

	 // Gradient Y
	 Sobel( image, gradY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	 convertScaleAbs( gradY, absGradY );

	 // Total Gradient (approximate)
	 addWeighted( absGradX, 0.5, absGradY, 0.5, 0, image );
	 return image;
}

/*****************************************************************************************************************************/
Mat populateLabels(const unsigned int numberOfImages, const unsigned int numberOfClasses){
	/*Build the training Matrix's Labels for the SVM. Each line will be the label of the respective image in the TrainingMat.
		 * @param  const unsigned int numberOfImages
		 * @param const unsigned int numberOfClasses
		 *
		 */
	std::vector<int> labels;
	for (unsigned int i=0; i<numberOfClasses;i++){
			for (unsigned int j=0 ; j<numberOfImages;j++){
				labels.push_back(i);
			}
	}
	Mat labelsMat(numberOfImages*numberOfClasses, 1, CV_32SC1 , labels.data());
	return labelsMat;

}

/*****************************************************************************************************************************/
void printSupportVectors(const Ptr<SVM>& svm){
	/* Print the Support Vectors.
	 * @param Ptr<SVM> svm The SVM containing the Support Vectors to show
	 */
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	int thickness = -1;
	int lineType = 8;
	// Show support vectors
	thickness = 2;
	lineType  = 8;
	Mat sv = svm->getUncompressedSupportVectors();
	for (int i = 0; i < sv.rows; ++i){
		const float* v = sv.ptr<float>(i);
		circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(30, 128, 0), thickness, lineType);
	}
	imwrite("result.png", image);
	imshow("Uncompressed Support Vectors", image);

}
/*****************************************************************************************************************************/
