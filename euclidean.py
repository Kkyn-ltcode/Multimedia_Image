import math
import pandas as pd
import numpy as np

def euclidean_distance(feat_1, feat_2):
    distance = 0.0
    if feat_1.shape < feat_2.shape:
        feat_1 = np.pad(feat_1, (0, feat_2.shape[0] - feat_1.shape[0]), mode='constant', constant_values=0.0).astype(np.float16)
    elif feat_1.shape > feat_2.shape:
        feat_2 = np.pad(feat_2, (0, feat_1.shape[0] - feat_2.shape[0]), mode='constant', constant_values=0.0).astype(np.float16)
    for i in range(len(feat_1)):
        distance += (feat_1[i] - feat_2[i]) ** 2
    return math.sqrt(distance)

def find_similar(img_feature):
    data = np.load('image_features.npy')
    data = pd.DataFrame(data)
    distances = data.copy()[[0]]
    distances.rename(columns={0: 'id'}, inplace=True)
    distances['id'] = distances['id'].astype(int)
    distances['distance'] = 0.0

    for i in range(len(data)):
        distances.loc[i, 'distance'] = euclidean_distance(img_feature, data.iloc[i][1:].values)
    return distances