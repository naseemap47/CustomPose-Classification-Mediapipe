# CustomPose-Classification-Mediapipe
Creating a Custom pose classification using Mediapipe with help of OpenCV

<p align="center">
  <img src='https://miro.medium.com/max/434/1*iy_qNrpaHWkfJTZ3TrAuKA.png'/>
</p>

**Sample Video Output:**<br>
<p align="center">
  <img src='https://user-images.githubusercontent.com/88816150/189837009-a7344d98-d795-4bc4-b1fd-640e772221f7.gif' alt="animated" />
</p>

**Sample Image Output:**<br>

<div class="row">
  <div class="column">
    <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/chair.jpg">
    <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/cobra.jpg">
    <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/dog.jpg">
  </div>
  <div class="column">
  <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/tree.jpg">
  <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/warrior.jpg">
  </div>
</div>

# (Demo) Let's Get Started...
Using this Custom Pose Classification, I am going to Create a Yoga Pose Classification. Using **Yoga Poses Dataset**.
### Clone this Repository
```
git clone https://github.com/naseemap47/CustomPose-Classification-Mediapipe.git
cd CustomPose-Classification-Mediapipe
```
### Install Dependency
```
pip3 install -r requirements.txt
```

### 1.Download Dataset:
**Yoga Poses Dataset:**
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
Show Predicted Pose Class on Test Image or Video or Web-cam <br>
**To Save:**
 - `--save`: It will save Images (on **ImageOutput** Dir) or Videos ("**output.avi**")
```
python3 inference.py --model <path_to_model> \
                     --conf <model_prediction_confidence> \
                     --source <image or video or web-cam>

# to save
python3 inference.py --model <path_to_model> \
                     --conf <model_prediction_confidence> \
                     --source <image or video or web-cam> \
                     --save
```
Example:
```
python3 inference.py --model model.h5 --conf 0.75 --source data/test/image.jpg
python3 inference.py --model model.h5 --conf 0.75 --source data/test/video.mp4
python3 inference.py --model model.h5 --conf 0.75 --source 0  # web-cam

# to save
python3 inference.py --model model.h5 --conf 0.75 --source data/test/image.jpg --save
python3 inference.py --model model.h5 --conf 0.75 --source data/test/video.mp4 --save
python3 inference.py --model model.h5 --conf 0.75 --source 0 --save # web-cam
```
**To Exit Window - Press Q-key**

# Custom Pose Classification
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
According to your Class Names, Write Class Order <br>
**To Save:**
 - `--save`: It will save Images (on **ImageOutput** Dir) or Videos ("**output.avi**")
```
python3 inference.py --model <path_to_model> \
                     --conf <model_prediction_confidence> \
                     --source <image or video or web-cam> \
                     
# to save
python3 inference.py --model <path_to_model> \
                     --conf <model_prediction_confidence> \
                     --source <image or video or web-cam> \
                     --save
```
Show Predicted Pose Class on Test Image or Video or Web-cam

**To Exit Window - Press Q-key**
