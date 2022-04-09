# import packages
from pydoc import apropos
from PIL.Image import Image
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

def build_model(input_shape:tuple):

    # this function will build a model with 5 block containing 2 convolutional
    # filters , a batch normalization layer, a maxpooling layer and a dropout layer

    model = Sequential()    # initialize a sequential model

    model.add(Rescaling(1./255))
    model.add(RandomFlip("horizontal_and_vertical"))
    model.add(RandomRotation(0.2))

    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='conv1_1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), name = 'pool1_1'))
    model.add(Dropout(0.3, name = 'drop1_1'))
    
   
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), name = 'pool2_1'))
    model.add(Dropout(0.3, name = 'drop2_1'))
    
    
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), name = 'pool3_1'))
    model.add(Dropout(0.3, name = 'drop3_1'))


    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv4_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv4_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv4_3'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv4_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), name = 'pool4_1'))


    model.add(Dropout(0.3, name = 'drop5_1'))#Flatten and output
    model.add(Flatten(name = 'flatten'))
    model.add(Dense(5, activation='softmax', name = 'output'))# create model 

    return model



if __name__ == '__main__':
    
    train='./dataset/TRAIN'
    test='./dataset/TEST'
    batch_size=16
    epochs=50
    output='models/deep_cnn_tf'

    normalization_layer = tf.keras.layers.Rescaling(1./255)


    IMG_SIZE = (224, 224) # image height and width for training

    # prepare train dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(train,
                                                            shuffle=True,               # shuffle the train dataset
                                                            batch_size=batch_size,
                                                            image_size=IMG_SIZE,
                                                             label_mode='categorical') 

    # preparing the validation dataset
    validation_dataset = tf.keras.utils.image_dataset_from_directory(test,
                                                                    shuffle=True,
                                                                    batch_size=batch_size,
                                                                    image_size=IMG_SIZE,
                                                                    label_mode='categorical')

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))


    optimizer =  Adam(learning_rate=0.001, decay=1e-6)  # amdam optimizer

    model = build_model(input_shape=(224,224,3))  # get the model 

    # compile all the layers and prepare the model for training
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    # train model
    history = model.fit_generator(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )

    # save model
    
    model.save(output)
