from keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to sample image")

ap.add_argument("-m", "--model", type=str, required=True,
                help="path to saved .h5 model, eg: dir/model.h5")

args = vars(ap.parse_args())
path_to_img = args["image"]
path_saved_model = args["model"]

##############
torso_size_multiplier = 2.5
n_landmarks = 33
n_dimensions = 3
landmark_names = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky_1', 'right_pinky_1',
    'left_index_1', 'right_index_1',
    'left_thumb_2', 'right_thumb_2',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]
class_names = [
    'Chair', 'Cobra', 'Dog',
    'Tree', 'Warrior'
]
##############

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

col_names = []
for i in range(n_landmarks):
    name = mp_pose.PoseLandmark(i).name
    name_x = name + '_X'
    name_y = name + '_Y'
    name_z = name + '_Z'
    name_v = name + '_V'
    col_names.append(name_x)
    col_names.append(name_y)
    col_names.append(name_z)
    col_names.append(name_v)

# Load saved model
model = load_model(path_saved_model, compile=True)

# Load sample Image
img = cv2.imread(path_to_img)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = pose.process(img_rgb)
if result.pose_landmarks:
    lm_list = []
    for landmarks in result.pose_landmarks.landmark:
        # Preprocessing
        max_distance = 0
        lm_list.append(landmarks)
    center_x = (lm_list[landmark_names.index('right_hip')].x +
                lm_list[landmark_names.index('left_hip')].x)*0.5
    center_y = (lm_list[landmark_names.index('right_hip')].y +
                lm_list[landmark_names.index('left_hip')].y)*0.5

    shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                   lm_list[landmark_names.index('left_shoulder')].x)*0.5
    shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                   lm_list[landmark_names.index('left_shoulder')].y)*0.5

    for lm in lm_list:
        distance = math.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2)
        if(distance > max_distance):
            max_distance = distance
    torso_size = math.sqrt((shoulders_x - center_x) **
                           2 + (shoulders_y - center_y)**2)
    max_distance = max(torso_size*torso_size_multiplier, max_distance)

    pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, (landmark.y-center_y)/max_distance,
                  landmark.z/max_distance, landmark.visibility] for landmark in lm_list]).flatten())
    data = pd.DataFrame([pre_lm], columns=col_names)
    predict = model.predict(data)[0]
    print('predictions: ', predict)
    print('predicted Pose Class: ', class_names[predict.argmax()])

# Show Result
img = cv2.putText(
    img, f'{class_names[predict.argmax()]}',
    (40, 50), cv2.FONT_HERSHEY_PLAIN,
    2, (255, 0, 255), 2
)
cv2.imshow('Output Image', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
