import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Conv2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
import os
from tqdm import tqdm

labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

x_train = []
y_train = []

x_test = []
y_test = []

train_path = (os.path.join('C:','Training', ))
test_path = (os.path.join('C:','Testing', ))

img_size = 300

for i in labels:
    train_path = os.path.join('C:','Training',i)
    for j in tqdm(os.listdir(train_path)):
        img = cv2.imread(os.path.join(train_path,j))
        img = cv2.resize(img,(img_size, img_size))
        x_train.append(img)
        y_train.append(i)

for i in labels:
    test_path = os.path.join('C:','Testing',i)
    for j in tqdm(os.listdir(test_path)):
        img = cv2.imread(os.path.join(test_path,j))
        img = cv2.resize(img,(img_size, img_size))
        x_test.append(img)
        y_test.append(i)

x_train = (np.array(x_train).astype(np.float16))
x_test = (np.array(x_test).astype(np.float16))

train_labels_encoded = [0 if category == 'no_tumor' 
    else(1 if category == 'glioma_tumor' 
    else(2 if category=='meningioma_tumor' 
    else 3)) for category in list(y_train)]
test_labels_encoded = [0 if category == 'no_tumor' 
    else(1 if category == 'glioma_tumor' 
    else(2 if category=='meningioma_tumor' 
    else 3)) for category in list(y_test)]

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train, test_size=0.1,random_state=101)

new_y_train = []
for i in y_train:
    new_y_train.append(labels.index(i))
y_train = new_y_train
y_train = tf.keras.utils.to_categorical(y_train)

new_y_test = []
for i in y_test:
    new_y_test.append(labels.index(i))
y_test = new_y_test
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.Sequential(
        [
          tf.keras.layers.Conv2D(kernel_size=(5,5) ,filters=32, activation='relu', padding='same', use_bias = True),
          tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'),

          tf.keras.layers.Conv2D(kernel_size=(5,5) ,filters=32, activation='relu', padding='same', use_bias = True),
          tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'),

          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation ='relu'),
          tf.keras.layers.Dropout(rate = 0.68),
          tf.keras.layers.Dense(4, activation ='softmax')
  ])

model.compile(optimizer = 'Nadam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, validation_split = 0.1, epochs = 9, verbose = 1, batch_size = 32)

prediction = model.predict(x_test)
prediction = np.argmax(prediction, axis = 1)
new_y_test = np.argmax(y_test, axis = 1)

print(classification_report(new_y_test, prediction))
