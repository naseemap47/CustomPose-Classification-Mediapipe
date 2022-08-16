from keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
from utils import NormPoseLandmark, min_max
import pandas as pd


##############
torso_size_multiplier=2.5
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
    'Downdog', 'Goddess', 'plank',
    'tree', 'warrior2'
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
    col_names.append(name_x)
    col_names.append(name_y)
    col_names.append(name_z)

path_to_model = 'model_val_loss_1.96.h5'
model = load_model(path_to_model, compile=True)

path_to_img = 'dataset/downdog/00000004.jpg'
img = cv2.imread(path_to_img)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


result = pose.process(img_rgb)
if result.pose_landmarks:
    lm_list = []
    for lm in result.pose_landmarks.landmark:
        lm_list.append(lm.x)
        lm_list.append(lm.y)
        lm_list.append(lm.z)

    # Preprocessing
    landmarks = np.array(lm_list, np.float32).reshape([n_landmarks, n_dimensions])
    norm_landmarks = NormPoseLandmark(landmark_names, landmarks)
    landmarks = np.array(norm_landmarks, np.float32).reshape([1, 99])
    # print(landmarks[0])

    data = pd.DataFrame(landmarks, columns=col_names)
    data.to_csv('min_max_df.csv', encoding='utf-8', index=False)
    # data = min_max(data)
    predict = model.predict(data)[0]
    print('prediction: ', predict)
    # class_id = np.argmax(predict, axis = 1)
    print('Pose_Class: ', class_names[predict.argmax()])
    # full_lm_list.append(landmarks[0])
    # target_list.append(class_name)