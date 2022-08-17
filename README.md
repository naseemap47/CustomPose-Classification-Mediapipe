# YogaPose-Classification-Mediapipe
Creating a Yoga pose classification using Mediapipe with help of OpenCV

## Custom Pose Classification
```
git checkout custom
```
### 1.Take your Custom Pose Dataset
### 2.Create Landmark Dataset for each Classes
CSV file will be saved in **<path_to_save_csv>**
```
python3 poseLandmark_csv.py -i <path_to_data_dir> -o <path_to_save_csv>
```
### 3.Create DeepLearinng Model to predict Human Pose
Model will saved in **<path_to_save_model>** and Model Metrics saved in **metrics.png**
```
python3 poseModel.py -i <path_to_save_csv> -o <path_to_save_model>
```
### 4.Inference
Open **inference.py**

change **Line-43**: 
According to your Class Names, Write Class Order
```
python3 inference.py --model <path_to_model> --conf <model_prediction_confidence> --source <image or video or web-cam>
```
Show Predicted Pose Class on Test Image or Video or Web-cam

**To Exit Window - Press Q-key**
