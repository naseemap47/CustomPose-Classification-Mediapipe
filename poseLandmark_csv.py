import os
import cv2
import mediapipe as mp
import glob
import pandas as pd
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")

ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save csv file, eg: dir/data.csv")

args = vars(ap.parse_args())

path_data_dir = args["dataset"]
path_to_save = args["save"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class_list = os.listdir(path_data_dir)

col_names = []
for i in range(33):
    name = mp_pose.PoseLandmark(i).name
    name_x = name + '_X'
    name_y = name + '_Y'
    name_z = name + '_Z'
    col_names.append(name_x)
    col_names.append(name_y)
    col_names.append(name_z)

col_names = col_names + ['Class_Name']
# print(col_list)

full_lm_list = []
for class_name in class_list:
    path_to_class = os.path.join(path_data_dir, class_name)
    img_list = glob.glob(path_to_class + '/*.jpg') + \
        glob.glob(path_to_class + '/*.jpeg') + \
        glob.glob(path_to_class + '/*.png')

    # Read reach Images in the each classes
    for img in img_list:
        image = cv2.imread(img)
        if image is None:
            print(
                f'[ERROR] Error in reading {img} -- Skipping.....\n[INFO] Taking next Image')
            continue
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)
            if result.pose_landmarks:
                lm_list = []
                for lm in result.pose_landmarks.landmark:
                    lm_list.append(lm.x)
                    lm_list.append(lm.y)
                    lm_list.append(lm.z)
                lm_list.append(class_name)
                full_lm_list.append(lm_list)
            print(f'{os.path.split(img)[1]} Landmarks added Successfully')
    print(f'[INFO] {class_name} Successfully Completed')

print('[INFO] Landmarks from Dataset Successfully Completed')

data = pd.DataFrame(full_lm_list, columns=col_names)
data.to_csv(path_to_save, encoding='utf-8', index=False)
print(f'[INFO] Successfully Saved Landmarks data into {path_to_save}')
