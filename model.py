import cv2 as cv
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

TRAINING_DIR = "/content/drive/MyDrive/Dataset/train" # path to training set
training_datagen = ImageDataGenerator(                # here apply some real time augmentations to images
      rescale = 1./255,                               #sclale the size of image between 0-1
	  rotation_range=40,                              # rotated in range 0-40
      width_shift_range=0.2,                          #width shift
      height_shift_range=0.2,                         #height shift
      shear_range=0.2,
      zoom_range=0.2,                                 #zoom image 
      horizontal_flip=True,
      fill_mode='nearest')                            #fill the rest with nearest pixel

VALIDATION_DIR = "/content/drive/MyDrive/Dataset/validation"  #path to validation set
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_datagenerator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (160,160),                                  #image resized to 100x100px
    class_mode = 'binary',                                    #2D one hot encoded
)


validation_datgenerator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size = (160,160),
    class_mode = 'binary'
)


#feature extracter : MobileNetV2

base_model = tf.keras.applications.MobileNetV2(weights="imagenet",           # trained from imagenet dataset
                                               include_top=False,            # to exclude output layer and dense layers
                                               input_shape=(160, 160, 3))    # input image shape



head_model = base_model.output                                              # connect with base model
head_model = tf.keras.layers.AveragePooling2D(pool_size=(5,5))(head_model)
head_model = tf.keras.layers.Flatten(name="flatten")(head_model)            # Flatten the array
head_model = tf.keras.layers.Dense(128,activation='relu')(head_model)
head_model = tf.keras.layers.Dropout(0.5)(head_model)                       # Introduce dropout to limit overfitting
head_model = tf.keras.layers.Dense(2,activation='softmax')(head_model)      # 2 neurons for 2 outputs   


base_model.trainable=False

model = tf.keras.models.Model(inputs = base_model.input , outputs = head_model)

model.summary()

lr = 1e-2 #learning rate


model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),   # uses the Adam optimizer try sgd and RMSprop
             loss = 'sparse_categorical_crossentropy', 
             metrics=['accuracy'])

history = model.fit(train_datagenerator, epochs = 20,validation_data =validation_datgenerator)

loss, accuracy = model.evaluate(validation_datgenerator)

model.save("Face_detector.h5") # save the model