import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras

print(keras.__version__)

import sklearn
samples = []

with open('p3-data-23/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Number of training samples = ', len(train_samples))
print('Number of validation samples = ', len(validation_samples))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = [] 
            measurements = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                correction = 0.3
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                image_center_path = 'p3-data-23/IMG/' + batch_sample[0]
                image_left_path = 'p3-data-23/IMG/' + batch_sample[1]
                image_right_path = 'p3-data-23/IMG/' + batch_sample[2]
                image = mpimg.imread(image_center_path)
                images.append(image)
                measurement = steering_center
                measurements.append(measurement)
                image = mpimg.imread(image_left_path)
                images.append(image)
                measurement = steering_left
                measurements.append(measurement)
                image = mpimg.imread(image_right_path)
                images.append(image)
                measurement = steering_right
                measurements.append(measurement)

            augmented_images, augmented_measurements = [], []
            for image,measurement in zip(images,measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
    
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

#plt.imshow(X_train[0])
#plt.show()

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Activation

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(6,(5,5), padding='valid'))  
#model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(2,2), padding='valid'))
model.add(Conv2D(16,(5,5), padding='valid'))  
#model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(2,2), padding='valid'))
model.add(Conv2D(32,(3,3), padding='valid'))  
model.add(MaxPooling2D(strides=(2,2), padding='valid'))
model.add(Conv2D(48,(3,3), padding='valid'))  
model.add(MaxPooling2D(strides=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(1164))
#model.add(Activation('relu'))
model.add(Dense(100))
#model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/32
        , validation_data = validation_generator, validation_steps = len(validation_samples)/32,
        epochs=5, verbose=1)

model.save('model.h5')

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()






