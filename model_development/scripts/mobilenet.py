# import packages
from pydoc import apropos
from PIL.Image import Image
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import matplotlib.pyplot as plt

def build_model(input_shape:tuple):

    base_model = MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False

    global_average_layer = GlobalAveragePooling2D()

    prediction_layer = Dense(5, activation='softmax')
   
    model = Sequential()    

    model.add(base_model)
    model.add(global_average_layer)
    model.add(prediction_layer)
    
    return model



if __name__ == '__main__':
    
    train='./dataset/TRAIN'
    test='./dataset/TEST'
    batch_size=16
    epochs=20
    output='models/mobilenet_tf.h5'

    IMG_SIZE = (224, 224) 

    train_dataset = tf.keras.utils.image_dataset_from_directory(train,
                                                            shuffle=True,               # shuffle the train dataset
                                                            batch_size=batch_size,
                                                            image_size=IMG_SIZE,
                                                             label_mode='categorical') 

    validation_dataset = tf.keras.utils.image_dataset_from_directory(test,
                                                                    shuffle=True,
                                                                    batch_size=batch_size,
                                                                    image_size=IMG_SIZE,
                                                                    label_mode='categorical')

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

    optimizer =  Adam(learning_rate=0.001, decay=1e-6)  

    model = build_model(input_shape=(224,224,3))  

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    history = model.fit_generator(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )

    model.save(output)


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy for mobilenet')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss for mobilenet')
    plt.legend()

    plt.show()

    
