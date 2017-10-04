# About 

This code shows the OpenCV's implementation of Support Vector Machines (SVM). I implemented and trained it to classify benign and malignant Melanoma images. There's a post on my [blog about it.](http://jeanvitor.com/opencv-svm-support-vector-melanoma/)

If you'd like, it's possible to train using also another kind of pictures.

# Compiling
  You will need OpenCV with [contrib modules](http://jeanvitor.com/cpp-opencv-windonws10-installing/).


# Results 

I selected 26 random images from  [International Society for Digital Imaging of the Skin (ISDIS) database](http://isdis.net/isic-project/), 13 from benign and 13 malignant lesions. From that, 20 were used in the training process and 6 (3 malignant and 3 benign) in the validation.
The SVM achieved the expected values in 100% of the cases.
