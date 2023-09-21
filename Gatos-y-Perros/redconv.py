
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import mlflow


ih, iw = 150, 150 #tamano de la imagen
input_shape = (ih, iw, 3) #forma de la imagen: alto ancho y numero de canales

#train_dir = 'data/minitrain' #directorio de entrenamiento
#test_dir = 'data/minitest' #directorio de prueba
train_dir = 'data/train' #directorio de entrenamiento
test_dir = 'data/test' #directorio de prueba


num_class = 2 #cuantas clases
epochs = 10 #cuantas veces entrenar. En cada epoch hace una mejora en los parametros

batch_size = 50 #batch para hacer cada entrenamiento. Lee 50 'batch_size' imagenes antes de actualizar los parametros. Las carga a memoria
num_train = 1200 #numero de imagenes en train
num_test = 1000 #numero de imagenes en test


epoch_steps = num_train // batch_size
test_steps = num_test // batch_size


gentrain = ImageDataGenerator(rescale=1. / 255.) #indica que reescale cada canal con valor entre 0 y 1.


train = gentrain.flow_from_directory(train_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')

gentest = ImageDataGenerator(rescale=1. / 255)

test = gentest.flow_from_directory(test_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')

#para cargar pesos de la red desde donde se quedó la ultima vez
#filename = "cvsd.h5"
#model.load_weights(filename)  #comentar si se comienza desde cero.
###

model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=(ih, iw,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(10, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



with mlflow.start_run() as run:
	model.fit_generator(
                train,
                steps_per_epoch=epoch_steps,
                epochs=epochs,
                validation_data=test,
                validation_steps=test_steps
                )


model.save('cvsd.h5')
