
# coding: utf-8

# In[ ]:


import pandas as pd
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Conv2D, PReLU, BatchNormalization, MaxPooling2D, Dropout, Flatten
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model, Sequential
from tqdm import tqdm_notebook
from tqdm import tqdm
from keras.optimizers import Adam
from keras import optimizers

import sys 
from PIL import Image 
import numpy  as np
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import gc


# In[ ]:


df = pd.read_csv('E:/Datasets/validation-annotations-human-imagelabels.csv', usecols=[0,2,3])
df = df[df.Confidence == 1]
classes = np.array(['/m/01g317', '/m/09j2d', '/m/04yx4', '/m/0dzct', '/m/07j7r', '/m/05s2s', '/m/03bt1vf', '/m/07yv9', '/m/0cgh4', '/m/01prls', '/m/09j5n', '/m/0jbk', '/m/0k4j', '/m/05r655', '/m/02wbm', '/m/0c9ph5', '/m/083wq', '/m/0c_jw', '/m/03jm5', '/m/0d4v4'])

li = []
for i in classes:
    li.append(df[df.LabelName == i])

df = pd.concat(li).sample(frac=1).reset_index(drop=True)

labels = df.LabelName.tolist()
Imageid = df.ImageID.values

#LightAug = Compose([
#      CLAHE(),
#      HorizontalFlip(.5),
#      ShiftScaleRotate(shift_limit=0.075, scale_limit=0.15, rotate_limit=10, p=.75 ),
#      Blur(blur_limit=3, p=.33),
#      OpticalDistortion(p=.33),
#      GridDistortion(p=.33),
#      HueSaturationValue(p=.33)
#  ], p=p)

classes = classes.tolist()
Y_train, Y_Val = [], []
for i in tqdm(labels[10000:20000]):
    temp = np.zeros(20)
    temp[classes.index(i)] = 1
    Y_train.append(temp)
    del temp
for i in tqdm(labels[:2000]):
    temp = np.zeros(20)
    temp[classes.index(i)] = 1
    Y_Val.append(temp)
    del temp
Y_train = np.array(Y_train)
Y_Val = np.array(Y_Val)
gc.collect(), Y_train[0]

