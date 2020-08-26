import os
import ast
import pandas as pd
from tqdm import tqdm
import numpy as np
import shutil

from sklearn.model_selection import train_test_split


DATA_PATH = "/home/hasan/Downloads/global-wheat-detection"
OUTPUT_PATH = "/home/hasan/Downloads/wheat_data"

df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
df.bbox = df.bbox.apply(ast.literal_eval)
df = df.groupby('image_id')["bbox"].apply(list).reset_index(name='bboxes')


df_train, df_valid = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)


df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)


def process_data(data, data_type='train'):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row['image_id']
        bounding_boxes = row['bboxes']
        yolo_data = []
        for bbox in bounding_boxes:
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]

            x_center = x + w / 2
            y_center = y + h / 2
            x_center /= 1024.0
            y_center /= 1024.0
            w /= 1024.0
            h /= 1024.0
            yolo_data.append([0, x_center, y_center, w, h])

        yolo_data = np.array(yolo_data)
        np.savetxt(
            os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name}.txt"),
            yolo_data,
            fmt=["%d", "%f", "%f", "%f", "%f"])
        shutil.copyfile(
            os.path.join(DATA_PATH, f"train/{image_name}.jpg"),
            os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg")
        )



process_data(df_train, data_type='train')
process_data(df_valid, data_type='validation')



train, wheat_data/images/train
val, wheat_data/images/validation
nc, 1
names, ["wheat"]
