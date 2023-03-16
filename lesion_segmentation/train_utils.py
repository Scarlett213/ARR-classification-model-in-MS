# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 19:07:34 2022

@author: Administrator
"""

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2 as cv
from tensorflow.keras import backend as K
import shutil
from model import creatcnn
def pick(index, X):
    a = []
    for i in range(len(X)):
        if i in index:
            a.append(X[i])
    return a

def index_title(index):
    title = ''
    for testId in index[:-1]:
        title = title + testId + '_'
    title = title + index[-1]
    return title

def checkDir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)
    else:
        return path


def movepng_test2train(path_train, path_test):
    # Move the pictures in the test folder to the train folder
    for modality in os.listdir(path_test):
        path_modality = os.path.join(path_test, modality)

        for img in os.listdir(path_modality):
            src = os.path.join(path_modality, img)
            dst = os.path.join(path_train, modality, img)
            shutil.move(src, dst)


def movepng_train2test(path_train, path_test, index):
    # Move the pictures in the train folder to the test folder
    for modality in os.listdir(path_train):
        for img in os.listdir(os.path.join(path_train, modality)):
            if (img.split('_')[0]) in index:
                src = os.path.join(path_train, modality, img)
                dst = os.path.join(path_test, modality, img)
                shutil.move(src, dst)

def GEN2D(path_train, batch_size, target_size):
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         validation_split=0.20,
                         horizontal_flip=True,
                         fill_mode='nearest',
                         rescale=1. / 255)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    path_train = path_train
    image_folder = 'MR'
    mask_folder = 'MASK'
    PETimage_folder = 'PET'
    batch_size = batch_size
    target_size = target_size
    image_generator_train = image_datagen.flow_from_directory(
        path_train,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        subset='training',
        target_size=target_size,
        seed=1)
    mask_generator_train = mask_datagen.flow_from_directory(
        path_train,
        classes=[mask_folder],
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        subset='training',
        target_size=target_size,
        seed=1)
    image_generator_val = image_datagen.flow_from_directory(
        path_train,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        subset='validation',
        target_size=target_size,
        seed=1)
    mask_generator_val = mask_datagen.flow_from_directory(
        path_train,
        classes=[mask_folder],
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        subset='validation',
        target_size=target_size,
        seed=1)
    PETimage_generator_train = image_datagen.flow_from_directory(
        path_train,
        classes=[PETimage_folder],
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        subset='training',
        target_size=target_size,
        seed=1)
    PETimage_generator_val = image_datagen.flow_from_directory(
        path_train,
        classes=[PETimage_folder],
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        subset='validation',
        target_size=target_size,
        seed=1)
    return image_generator_train, image_generator_val, PETimage_generator_train, PETimage_generator_val, mask_generator_train, mask_generator_val


def generate_batch(x_train, x_train2, y_train, randomFlag=True):
    while True:
        yield ({'MRinputs': x_train.next(), 'PETinputs': x_train2.next()}, {'output': y_train.next()})

def Loss_Graph(history, title, pngname):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.title(title)
    plt.savefig(pngname)
    plt.show()

def readImg(path):
    imgX = cv.imread(path, 0)
    imgX = imgX / 255.0
    imgX = np.reshape(imgX, imgX.shape + (1,))
    imgX = np.reshape(imgX, (1,) + imgX.shape)
    return imgX


def testGenerator(path_test):
    path_mr = os.path.join(path_test, 'MR')
    path_pet = os.path.join(path_test, 'PET')
    for i in os.listdir(path_mr):
        path_imgMr = os.path.join(path_mr, i)
        path_imgPet = os.path.join(path_pet, i)
        imgX = readImg(path_imgMr)
        imgY = readImg(path_imgPet)
        yield ({'MRinputs': imgX, 'PETinputs': imgY})

def labelVisualize(img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out=img
    img_out=255*img_out
    return img_out


def saveResult(path_save,path_test,npyfile):
    import shutil
    shutil.rmtree(path_save)
    os.mkdir(path_save)
    for i,item in enumerate(npyfile):
            img = labelVisualize(item)
            cv.imwrite(os.path.join(path_save,os.listdir(path_test)[i]),img)

def MyTrainAndTest(lr, input_size, steps_per_epoch, validation_steps, epochs, title, hdfname, pngname, path_test,
                   test_amount,
                   min_delta, min_lr, pe, pr,
                   image_generator_train, image_generator_val, PETimage_generator_train, PETimage_generator_val,
                   mask_generator_train, mask_generator_val):
    K.clear_session()
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            sess.run(tf.global_variables_initializer())  # tf.initialize_all_variables())
            sess.run(tf.local_variables_initializer())
            model = creatcnn(input_size)
            model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
                          metrics=['accuracy'])
            model_checkpoint = ModelCheckpoint(hdfname, monitor='val_loss', verbose=1, save_best_only=True)
            model_earlystopping = EarlyStopping(monitor='val_loss', patience=pe, verbose=0, mode='min')
            model_reduceLR = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=pr, verbose=0, mode="min",
                                               min_delta=min_delta, cooldown=0, min_lr=min_lr)
            history = model.fit_generator(generator=generate_batch(image_generator_train,
                                                                   PETimage_generator_train, mask_generator_train),
                                          steps_per_epoch=steps_per_epoch,
                                          validation_data=generate_batch(image_generator_val, PETimage_generator_val,
                                                                         mask_generator_val),
                                          validation_steps=validation_steps,
                                          callbacks=[model_reduceLR, model_earlystopping, model_checkpoint],
                                          epochs=epochs)
            Loss_Graph(history, title, pngname)
            # test
            testgen = testGenerator(path_test)
            result = model.predict_generator(testgen, test_amount, verbose=1)
            return result
