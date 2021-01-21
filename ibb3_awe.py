from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import scipy as sp

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)
train_dataset = train.flow_from_directory('awe_sample/train/',
                                          target_size= (200,200),
                                          batch_size= 10,
                                          class_mode= 'binary')
validation_dataset = validation.flow_from_directory('awe_sample/val/',
                                          target_size= (200,200),
                                          batch_size= 5,
                                          class_mode= 'binary')
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('accuracy') > .98) & (logs.get('val_accuracy') > .9):
            print("Reached 98% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()

model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
     tf.keras.layers.MaxPool2D(),
     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
     tf.keras.layers.MaxPool2D(),
     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
     tf.keras.layers.MaxPool2D(),
     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
     tf.keras.layers.MaxPool2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512,activation='relu'),
     tf.keras.layers.Dense(1,activation='sigmoid')
     ])

model.compile(loss="binary_crossentropy",
              optimizer= RMSprop(lr=0.001),
              metrics=['accuracy'])
model_filt = model.fit(train_dataset,
                       steps_per_epoch=3,
                       epochs=35,
                       validation_data=validation_dataset,
                       callbacks=[callbacks])


dir_path = 'awe_sample/awe_test/'

sez = []
for i in os.listdir(dir_path):
    ime = i.split(".")
    cifra = ime[0]
    if cifra[0] == "0":
        sez.append("male")
    elif cifra[0] == "1":
        sez.append("female")
    else:
        sez.append("Unknown")

#print(sez)

correct = 0
rez = []
for i in os.listdir(dir_path):
   #print(i)
    img = image.load_img(dir_path+'//'+i, target_size=(200, 200))
    #plt.imshow(img)
    #plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    classes = model.predict(images)
    #print(classes[0])

    if classes[0] == 0:
        print("female")
        rez.append("female")
    elif classes[0] == 1:
        print("male")
        rez.append("male")
    else:
        print("Unknown")
        rez.append("Unknown")

ind = 0
for im in rez:
    if rez[ind] == sez[ind]:
        correct = correct+1
    ind = ind+1

ratio = correct / len(sez)
print("The ratio of correct predictions is", ratio)