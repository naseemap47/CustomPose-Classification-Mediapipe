import os
import cv2
import mediapipe as mp
import glob
import pandas as pd
import argparse
import numpy as np
import math


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")

ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save csv file, eg: dir/data.csv")

args = vars(ap.parse_args())

path_data_dir = args["dataset"]
path_to_save = args["save"]

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
##############

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class_list = os.listdir(path_data_dir)
class_list = sorted(class_list)

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

full_lm_list = []
target_list = []
for class_name in class_list:
    path_to_class = os.path.join(path_data_dir, class_name)
    img_list = glob.glob(path_to_class + '/*.jpg') + \
        glob.glob(path_to_class + '/*.jpeg') + \
        glob.glob(path_to_class + '/*.png')
    img_list = sorted(img_list)

    # Read reach Images in the each classes
    for img in img_list:
        image = cv2.imread(img)
        h, w, c = image.shape
        if image is None:
            print(
                f'[ERROR] Error in reading {img} -- Skipping.....\n[INFO] Taking next Image')
            continue
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                    distance = math.sqrt(
                        (lm.x - center_x)**2 + (lm.y - center_y)**2)
                    if(distance > max_distance):
                        max_distance = distance
                torso_size = math.sqrt(
                    (shoulders_x - center_x)**2 + (shoulders_y - center_y)**2)
                max_distance = max(
                    torso_size*torso_size_multiplier, max_distance)

                pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, (landmark.y-center_y)/max_distance,
                              landmark.z/max_distance, landmark.visibility] for landmark in lm_list]).flatten())

                full_lm_list.append(pre_lm)
                target_list.append(class_name)

            print(f'{os.path.split(img)[1]} Landmarks added Successfully')
    print(f'[INFO] {class_name} Successfully Completed')

print('[INFO] Landmarks from Dataset Successfully Completed')

data_x = pd.DataFrame(full_lm_list, columns=col_names)
data = data_x.assign(Pose_Class=target_list)
data.to_csv(path_to_save, encoding='utf-8', index=False)
print(f'[INFO] Successfully Saved Landmarks data into {path_to_save}')
