import pandas as pd
from keras import layers, callbacks, Sequential
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to csv Data")

ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save .h5 model, eg: dir/model.h5")

args = vars(ap.parse_args())
path_csv = args["dataset"]
path_to_save = args["save"]


# Load .csv Data
df = pd.read_csv(path_csv)
class_list = df['Pose_Class'].unique()
class_list = sorted(class_list)
class_number = len(class_list)

# Create training and validation splits
df['Pose_Class'], _ = df['Pose_Class'].factorize()
df_train = df.sample(frac=0.8, random_state=0)
df_valid = df.drop(df_train.index)


# # Scale to [0, 1]
# max_ = df_train.max(axis=0)
# min_ = df_train.min(axis=0)
# df_train = (df_train - min_) / (max_ - min_)
# df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
x_train = df_train.drop('Pose_Class', axis=1)
x_test = df_valid.drop('Pose_Class', axis=1)
y_train = df_train['Pose_Class']
y_test = df_valid['Pose_Class']

print('[INFO] Loaded csv Dataset')

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    min_delta=0.0001,  # minimium amount of change to count as an improvement
    patience=30,  # how many epochs to wait before stopping
    restore_best_weights=True,
    verbose=1
)

model = Sequential([
    layers.Dense(128, activation='relu', input_shape=[99]),
    # layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    # layers.Dense(512, activation='relu'),
    # layers.Dense(1024, activation='relu'),
    # layers.Dense(2048, activation='relu'),
    # layers.Dense(4096, activation='relu'),
    # layers.Dense(4096, activation='relu'),
    # layers.Dense(2048, activation='relu'),
    # layers.Dense(1024, activation='relu'),
    layers.Dense(128, activation='relu'),
    # layers.Dense(class_number),
    layers.Dense(class_number)
])

# Model Summary
print(model.summary())

model.compile(
    optimizer='adam',
    loss='mae',
)

print('[INFO] Model Training Started ...')
# Train the Model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=500,
    callbacks=[early_stopping],  # put your callbacks in a list
    verbose=1,  # turn off training log
)
print('[INFO] Model Training Completed')
val_loss = history.history['val_loss'][-1]
save_dir = os.path.split(path_to_save)[0]
model_name = os.path.split(path_to_save)[1]
model_name = os.path.splitext(model_name)[0] + f'_val_loss_{val_loss:.3}.h5'
path_to_save = os.path.join(save_dir, model_name)
model.save(path_to_save)
print(f'[INFO] Successfully Saved {path_to_save}')
