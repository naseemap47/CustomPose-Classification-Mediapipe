# CustomPose-Classification-Mediapipe
Creating a Custom pose classification using Mediapipe with help of OpenCV

<p align="center">
  <img src='https://miro.medium.com/max/434/1*iy_qNrpaHWkfJTZ3TrAuKA.png'/>
</p>


## Demo
**(Yoga Poses Dataset)**


Using this Custom Pose Classification, I created a Yoga Pose Classification
### Clone this Repository
```
git clone https://github.com/naseemap47/CustomPose-Classification-Mediapipe.git
cd CustomPose-Classification-Mediapipe
```
### 1.Download Dataset: 
```
wget -O yoga_poses.zip http://download.tensorflow.org/data/pose_classification/yoga_poses.zip
```
About Dataset:
- 5 Classes: **Chair, Cobra, Dog, Tree and Warrior**
- Contain Train and Test data
- Combain both Train and Test data

**Dataset Structure:**
```
├── Dataset
│   ├── Chair
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── Cobra
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
.   .
.   .
```

### 2.Create Landmark Dataset for each Classes
```
python3 poseLandmark_csv.py -i <path_to_data_dir> -o <path_to_save_csv>
```
Example:
```
python3 poseLandmark_csv.py -i data/ -o data.csv
```
CSV file will be saved in **<path_to_save_csv>**
### 3.Create DeepLearinng Model to predict Human Pose
```
python3 poseModel.py -i <path_to_save_csv> -o <path_to_save_model>
```
Example:
```
python3 poseModel.py -i data.csv -o model.h5
```
Model will saved in **<path_to_save_model>** and Model Metrics saved in **metrics.png**
### 4.Inference
Show Predicted Pose Class on Test Image or Video or Web-cam
```
python3 inference.py --model <path_to_model> --conf <model_prediction_confidence> --source <image or video or web-cam>
```
Example:
```
python3 inference.py --model model.h5 --conf 0.75 --source data/test/image.jpg
python3 inference.py --model model.h5 --conf 0.75 --source data/test/video.mp4
python3 inference.py --model model.h5 --conf 0.75 --source 0  # web-cam
```
**To Exit Window - Press Q-key**

## Custom Pose Classification
### Clone this Repository
```
git clone https://github.com/naseemap47/CustomPose-Classification-Mediapipe.git
cd CustomPose-Classification-Mediapipe
git checkout custom
```
### 1.Take your Custom Pose Dataset
**Dataset Structure:**
```
├── Dataset
│   ├── Pose_1
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── Pose_2
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
.   .
.   .
```
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
