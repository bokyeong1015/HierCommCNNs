# HierCommDeepCNNs_MATLAB_revised
MATLAB codes for
B.-K. Kim, H. Lee, J. Roh, and S.-Y. Lee (2015), 
"Hierarchical committee of deep cnns with exponentially-weighted decision fusion for static facial expression recognition."
In Proceedings of the 2015 ACM on International Conference on Multimodal Interaction (pp. 427-434). ACM.

------------------------------------------------------

Folder1: lib1_AlignFace_NormalizeInput

- codes used in face registration (multi-pipeline-based alignment)
- codes used for input normalization (illumination normalization, contrast enhancement)
         + formation of input matrix (imdb) for MatConvNet toolbox 

* Due to redistribution issues, we cannot provide some open-source codes used for our face registration.

- Please download the following libraries and locate them to the correct folders

1) /pipeline_modules_functions/module1_ZR_FaceDetector
-  visit https://www.ics.uci.edu/~xzhu/face/ -> download and unzip "face-release1.0-basic.zip" -> move the files in "face-release1.0-basic.zip" to "module1_ZR_FaceDetector"
	
2) /pipeline_modules_functions/module3_INTRAFACE_LandmarkDetector
- visit http://humansensing.cs.cmu.edu/intraface/download_functions_matlab.html -> download and unzip "FacialFeatureDetection&Tracking_v1.4.0.zip" -> move the files in "FacialFeatureDetection&Tracking_v1.4.0.zip" to "module3_INTRAFACE_LandmarkDetector"
	
(These days the IntraFace matlab codes '2)' are not found @ http://humansensing.cs.cmu.edu/intraface/download_functions_matlab.html
Please contact IntraFace team for downloading "FacialFeatureDetection&Tracking_v1.4.0.zip")

======================================================
Folder2: lib2_TrainDeepCNN

- codes used for training deep CNNs
  : based on MatConvNet toolbox (version1.0-beta8)

======================================================
Folder3: lib3_HierarchicalCommittee

- codes used in formation of hierarchical committee of deep CNNs
  : single-level committee, two-level committee


------------------------------------------------------

Notice that due to memory and license issues, we cannot provide the databases (SFEW2.0 + external data (FER-2013, TFD)), the trained models (240 deep CNNs specified in our article), and thus the complete codes for our work.

However, here we provide executable codes working on some sample data.
If you get all databases, fully understand our codes, and properly modify these codes for working on whole data and for training whole deep CNNs, you can obtain experimental results presented in our work.

If you have any query, please contact Bo-Kyeong Kim (bokyeong1015@gmail.com).
