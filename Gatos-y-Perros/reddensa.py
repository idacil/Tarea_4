import os
from glob import glob
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


#pip install sklearn
from sklearn.model_selection import train_test_split
from PIL import Image

import mlflow

mlflow.tensorflow.autolog()

traincat = 'data/minitrain/cat/'
traindog = 'data/minitrain/dog/'

cat_files_path = os.path.join(traincat, '*')
dog_files_path = os.path.join(traindog, '*')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))

n_files = len(cat_files) + len(dog_files)
print(n_files)

size_image = 64

allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0
for f in cat_files:
    try:
        img = Image.open(f)
        new_img = img.resize(size=(size_image, size_image))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        print("No cargo imagen")
        #continue

for f in dog_files:
    try:
        img = Image.open(f)
        new_img = img.resize(size=(size_image, size_image))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue


x, x_test, y, y_test = train_test_split(allX, ally, test_size=0.2, random_state=1)


print(x.shape[0])
print(x_test.shape)
print(y.shape)
print(y_test.shape)


x_train = x.reshape(x.shape[0], 64*64*3)
x_test = x_test.reshape(x_test.shape[0], 64*64*3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255.
x_test /= 255.

y_train = y
print(y_train)
print(y_train.shape)
print("-----")
print(x_train.shape)


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(64*64*3,)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(learning_rate=0.0001),metrics=['accuracy'])

history = model.fit(x_train, y_train,batch_size=10,epochs=10,verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print(score)
