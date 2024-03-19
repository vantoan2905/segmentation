import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  

import nilearn as nl 
import nibabel as nib 
import nilearn.plotting as nlplt 

import gif_your_nifti.core as gif2nif 

import keras
import keras.backend as K
from keras.layers import Input, Flatten
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import preprocessing


np.set_printoptions(precision=3, suppress=True)

SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', 
    2 : 'EDEMA',
    3 : 'ENHANCING' 
}

VOLUME_SLICES = 100 
VOLUME_START_AT = 22 
IMG_SIZE=128
TRAIN_DATASET_PATH = os.path.join('E:\\learn\\nkkh\\detetection\\dataset\\for_nifti_sengmentation\\data_brast_2020\\data\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData/')
VALIDATION_DATASET_PATH = os.path.join('E:\\learn\\nkkh\\detetection\\dataset\\for_nifti_sengmentation\\data_brast_2020\\data\\BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData/')

class Metrics:
    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1.0):
        """Computes Dice Coefficient (DC) of y_true and y_pred for each class.

        It computes the DC as the average of DCs of all classes.

        Args:
            y_true: 4D one hot encoding of true segmentation masks.
            y_pred: 4D probability maps of predicted segmentation masks.
            smooth: Smoothing parameter for computing DC.

        Returns:
            DC of y_true and y_pred over all classes.
        """
        class_num = 4
        # Flatten the first 3 dimensions of y_true and y_pred
        for i in range(class_num):
            y_true_f = tf.keras.backend.flatten(y_true[:, :, :, i])
            y_pred_f = tf.keras.backend.flatten(y_pred[:, :, :, i])
            # Compute intersection and loss
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            loss = ((2. * intersection + smooth) 
                    / (tf.keras.backend.sum(y_true_f) 
                       + tf.keras.backend.sum(y_pred_f) + smooth))
            # Print loss of each class
            tf.keras.backend.print_tensor(loss, message='loss value for class {}: '.format(SEGMENT_CLASSES[i]))
            # Add loss of each class to total loss
            if i == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        # Compute average loss over all classes
        total_loss = total_loss / class_num
        return total_loss

    @staticmethod
    def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
        return (2. * intersection) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:, :, :, 1])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:, :, :, 1])) + epsilon)

    @staticmethod
    def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
        return (2. * intersection) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:, :, :, 2])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:, :, :, 2])) + epsilon)

    @staticmethod
    def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:, :, :, 3] * y_pred[:, :, :, 3]))
        return (2. * intersection) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:, :, :, 3])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:, :, :, 3])) + epsilon)

    @staticmethod
    def precision(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    @staticmethod
    def sensitivity(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + tf.keras.backend.epsilon())

    @staticmethod
    def specificity(y_true, y_pred):
        true_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + tf.keras.backend.epsilon())


class UNetBuilder:
    """
    Build U-Net model.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor.
    ker_init : str, optional
        Kernel initializer, by default 'he_normal'
    dropout : float, optional
        Dropout rate, by default 0.2

    """
    def __init__(self, inputs, ker_init='he_normal', dropout=0.2):
        """
        Initialize U-Net builder.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        ker_init : str, optional
            Kernel initializer, by default 'he_normal'
        dropout : float, optional
            Dropout rate, by default 0.2

        """
        self.inputs = inputs
        """Input tensor."""

        self.ker_init = ker_init
        """Kernel initializer."""

        self.dropout = dropout
        """Dropout rate."""

    def build_unet(self):
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(self.inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv1)

        pool = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(pool)
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv3)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(pool4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv5)
        drop5 = Dropout(self.dropout)(conv5)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=self.ker_init)(
            UpSampling2D(size=(2, 2))(drop5))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=self.ker_init)(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=self.ker_init)(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv9)

        up = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=self.ker_init)(
            UpSampling2D(size=(2, 2))(conv9))
        merge = concatenate([conv1, up], axis=3)
        conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(merge)
        conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv)

        conv10 = Conv2D(4, (1, 1), activation='softmax')(conv)

        return Model(inputs=self.inputs, outputs=conv10)



class DataGenerator(keras.utils.Sequence):
    """
    Data generator for data augmentation and batching.

    Parameters
    ----------
    list_IDs : list
        List of patient IDs to use
    dim : tuple, optional
        Size of the input data (height, width), by default (IMG_SIZE,IMG_SIZE)
    batch_size : int, optional
        Batch size, by default 1
    n_channels : int, optional
        Number of channels in the input data, by default 2
    shuffle : bool, optional
        Whether to shuffle the data at the end of each epoch, by default True
    """
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Batch_ids = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii');
            flair = nib.load(data_path).get_fdata()    

            data_path = os.path.join(case_path, f'{i}_t1ce.nii');
            ce = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, f'{i}_seg.nii');
            seg = nib.load(data_path).get_fdata()
        
            for j in range(VOLUME_SLICES):
                 X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
                 X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

                 y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT];
                    
        y[y==4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE));
        return X/np.max(X), Y
        


