from model2 import Metrics, DataGenerator, UNetBuilder
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

def main():
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
    train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
    train_and_val_directories.remove(TRAIN_DATASET_PATH + 'BraTS20_Training_355')
    def pathListIntoIds(dirList):
        x = []
        for i in range(0, len(dirList)):
            x.append(dirList[i][dirList[i].rfind('/') + 1:])
        return x
    train_and_test_ids = pathListIntoIds(train_and_val_directories)
    train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2)
    train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15)
    training_generator = DataGenerator(train_ids) 
    valid_generator = DataGenerator(val_ids)
    test_generator = DataGenerator(test_ids)
    csv_logger = CSVLogger('training.log', separator=',', append=False)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1, mode='auto'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.weights.h5', verbose=1, save_best_only=True, save_weights_only = True),
        csv_logger
    ]
    input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
    model_builder = UNetBuilder(input_layer)
    model = model_builder.build_unet()
    
    callbacks = [ModelCheckpoint("model_checkpoint.h5", save_best_only=True)]
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=4), Metrics.dice_coef, Metrics.precision, Metrics.sensitivity, Metrics.specificity, Metrics.dice_coef_necrotic, Metrics.dice_coef_edema, Metrics.dice_coef_enhancing])
    hist = model.fit(training_generator, epochs=3, steps_per_epoch=len(train_ids), callbacks=callbacks, validation_data=valid_generator)
    model.save("model_x1_1.h5")
    model = keras.models.load_model(
        # r'E:\\learn\\nkkh\\detetection\\sengmentation\\model_x1_1.h5',
        './model_x1_1.h5',
        custom_objects={
            'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),    
            'dice_coef': Metrics.dice_coef,
            'precision': Metrics.precision,
            'sensitivity': Metrics.sensitivity,
            'specificity': Metrics.specificity,
            'dice_coef_necrotic': Metrics.dice_coef_necrotic,
            'dice_coef_edema': Metrics.dice_coef_edema,
            'dice_coef_enhancing':Metrics.dice_coef_enhancing
        },
        compile=False
    )
    hist = pd.read_csv(
        './training.log',
        sep=',',
        engine='python'
    )
    epoch = hist['epoch']
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    train_dice = hist['dice_coef']
    val_dice = hist['val_dice_coef']
    f, ax = plt.subplots(1, 4, figsize=(16, 8))
    ax[0].plot(epoch, acc, 'b', label='Training Accuracy')
    ax[0].plot(epoch, val_acc, 'r', label='Validation Accuracy')
    ax[0].legend()
    ax[1].plot(epoch, loss, 'b', label='Training Loss')
    ax[1].plot(epoch, val_loss, 'r', label='Validation Loss')
    ax[1].legend()
    ax[2].plot(epoch, train_dice, 'b', label='Training dice coef')
    ax[2].plot(epoch, val_dice, 'r', label='Validation dice coef')
    ax[2].legend()
    ax[3].plot(epoch, hist['mean_io_u'], 'b', label='Training mean IOU')
    ax[3].plot(epoch, hist['val_mean_io_u'], 'r', label='Validation mean IOU')
    ax[3].legend()
    plt.show()
    def imageLoader(path):
        image = nib.load(path).get_fdata()
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        for j in range(VOLUME_SLICES):
            X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(image[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
            X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
            y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT];
        return np.array(image)
    def loadDataFromDir(path, list_of_files, mriType, n_images):
        scans = []
        masks = []
        for i in list_of_files[:n_images]:
            fullPath = glob.glob( i + '/*'+ mriType +'*')[0]
            currentScanVolume = imageLoader(fullPath)
            currentMaskVolume = imageLoader( glob.glob( i + '/*seg*')[0] ) 
            for j in range(0, currentScanVolume.shape[2]):
                scan_img = cv2.resize(currentScanVolume[:,:,j], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype('uint8')
                mask_img = cv2.resize(currentMaskVolume[:,:,j], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype('uint8')
                scans.append(scan_img[..., np.newaxis])
                masks.append(mask_img[..., np.newaxis])
        return np.array(scans, dtype='float32'), np.array(masks, dtype='float32')
    def predictByPath(case_path,case):
        files = next(os.walk(case_path))[2]
        X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
        vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii');
        flair=nib.load(vol_path).get_fdata()
        vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii');
        ce=nib.load(vol_path).get_fdata() 
        for j in range(VOLUME_SLICES):
            X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
            X[j,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        return model.predict(X/np.max(X), verbose=1)
    def showPredictsById(case, start_slice = 60):
        path = f"E:\\learn\\nkkh\\detetection\\dataset\\for_nifti_sengmentation\\data_brast_2020\\data\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_{case}"
        gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
        origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
        p = predictByPath(path,case)
        core = p[:,:,:,1]
        edema= p[:,:,:,2]
        enhancing = p[:,:,:,3]
        plt.figure(figsize=(18, 50))
        f, axarr = plt.subplots(1,6, figsize = (18, 50)) 
        for i in range(6):
            axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
        axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
        axarr[0].title.set_text('Original image flair')
        curr_gt=cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
        axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3) 
        axarr[1].title.set_text('Ground truth')
        axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
        axarr[2].title.set_text('all classes')
        axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
        axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
        axarr[4].imshow(core[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
        axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
        axarr[5].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
        axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
        plt.show()
    showPredictsById(case=test_ids[0][-3:])
    showPredictsById(case=test_ids[1][-3:])
    showPredictsById(case=test_ids[2][-3:])
    showPredictsById(case=test_ids[3][-3:])
    showPredictsById(case=test_ids[4][-3:])
    showPredictsById(case=test_ids[5][-3:])
    showPredictsById(case=test_ids[6][-3:])
    case = case=test_ids[3][-3:]
    path = f"E:\\learn\\nkkh\\detetection\\dataset\\for_nifti_sengmentation\\data_brast_2020\\data\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    p = predictByPath(path,case)
    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]
    i=40 
    eval_class = 2 
    gt[gt != eval_class] = 1 
    resized_gt = cv2.resize(gt[:,:,i+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    plt.figure()
    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(resized_gt, cmap="gray")
    axarr[0].title.set_text('ground truth')
    axarr[1].imshow(p[i,:,:,eval_class], cmap="gray")
    axarr[1].title.set_text(f'predicted class: {SEGMENT_CLASSES[eval_class]}')
    plt.show()
    print("Evaluate on test data")
    
    results = model.evaluate(test_generator, batch_size=100, callbacks= callbacks)
    print("test loss, test acc:", results)

if __name__ == '__main__':
    main()