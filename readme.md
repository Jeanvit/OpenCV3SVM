# About 

This code shows the OpenCV's implementation of Support Vector Machines (SVM). I implemented and trained it to classify benign and malignant Melanoma images. There's a post on my [blog about it.](http://jeanvitor.com/opencv-svm-support-vector-melanoma/)

If you'd like, it's possible to train it using another kind of pictures.

# Compiling
  You will need OpenCV with [contrib modules](http://jeanvitor.com/cpp-opencv-windonws10-installing/).
  Considering the OpenCV folders `C:\\opencv-master\\mingw_build\\install\\include` and `C:\\opencv-master\\mingw_build\\install\\x86\\mingw\\lib`:
  
### G++
1. `cd src`

2. `g++ "-IC:\\opencv-master\\mingw_build\\install\\include" -O0 -g3 -Wall -c -fmessage-length=0 -std=c++14 -o "OPENCVSVM.o" "OPENCVSVM.cpp"` 

3. `g++ "-LC:\\opencv-master\\mingw_build\\install\\x86\\mingw\\lib" -o OPENCVSVM.exe "OPENCVSVM.o" -lopencv_calib3d330 -lopencv_imgcodecs330 -lopencv_imgproc330 -lopencv_ml330 -lopencv_objdetect330 -lopencv_photo330 -lopencv_shape330 -lopencv_core330 -lopencv_features2d330 -lopencv_highgui330 `


# Results 

I selected 26 random images from  [International Society for Digital Imaging of the Skin (ISDIS) database](http://isdis.net/isic-project/), 13 from benign and 13 malignant lesions. From that, 20 were used in the training process and 6 (3 malignant and 3 benign) in the validation.
The SVM achieved the expected values in 100% of the cases.
