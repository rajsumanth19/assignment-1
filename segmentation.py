import os
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

from config import config_segmentation
from config import segmentation_new_size
from plotdata import plot_segmentation_test


METRICS = [
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.Accuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.MeanIoU(num_classes=2, name='iou'),
    tf.keras.metrics.BinaryAccuracy(name='bin_accuracy'),
]


def segmentation_keras_load():
    
    batch_size = config_segmentation.get('batch_size')
    img_size = (segmentation_new_size.get("width"), segmentation_new_size.get("height"))
    img_width = img_size[0]
    img_height = img_size[1]
    epochs = config_segmentation.get('Epochs')
    img_channels = config_segmentation.get('CHANNELS')
    dir_images = "frames/Segmentation/Data/Images"
    dir_masks = "frames/Segmentation/Data/Masks"
    num_classes = config_segmentation.get("num_class")
    allfiles_image = sorted(
        [
            os.path.join(dir_images, fname)
            for fname in tqdm(os.listdir(dir_images))
            if fname.endswith(".jpg")
        ]
    )
    allfiles_mask = sorted(
        [
            os.path.join(dir_masks, fname)
            for fname in tqdm(os.listdir(dir_masks))
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    print("No of samples:", len(allfiles_image))
    for input_path, target_path in tqdm(zip(allfiles_image[:10], allfiles_mask[:10])):
        print(input_path, "|", target_path)
    total_samples = len(allfiles_mask)
    train_ratio = config_segmentation.get("train_set_ratio")
    val_samples = int(total_samples * (1 - train_ratio))
    random.Random(1337).shuffle(allfiles_image)
    random.Random(1337).shuffle(allfiles_mask)
    train_img_paths = allfiles_image[:-val_samples]
    train_mask_paths = allfiles_mask[:-val_samples]
    val_img_paths = allfiles_image[-val_samples:]
    val_mask_paths = allfiles_mask[-val_samples:]

    x_train = np.zeros((len(train_img_paths), img_height, img_width, img_channels), dtype=np.uint8)
    y_train = np.zeros((len(train_mask_paths), img_height, img_width, 1), dtype=np.bool)

    x_val = np.zeros((len(val_img_paths), img_height, img_width, img_channels), dtype=np.uint8)
    y_val = np.zeros((len(val_mask_paths), img_height, img_width, 1), dtype=np.bool)
    print('\nLoading training data: ', len(train_img_paths), 'images ...')
    for n, file_ in tqdm(enumerate(train_img_paths)):
        img = tf.keras.preprocessing.image.load_img(file_, target_size=img_size)
        x_train[n] = img

    print('\nLoading training masks: ', len(train_mask_paths), 'masks ...')
    for n, file_ in tqdm(enumerate(train_mask_paths)):
        img = tf.keras.preprocessing.image.load_img(file_, target_size=img_size, color_mode="grayscale")
        y_train[n] = np.expand_dims(img, axis=2)

    print('\nLoading test data: ', len(val_img_paths), 'images ...')
    for n, file_ in tqdm(enumerate(val_img_paths)):
        img = tf.keras.preprocessing.image.load_img(file_, target_size=img_size)
        x_val[n] = img

    print('\nLoading test masks: ', len(val_mask_paths), 'masks ...')
    for n, file_ in tqdm(enumerate(val_mask_paths)):
        img = tf.keras.preprocessing.image.load_img(file_, target_size=img_size, color_mode="grayscale")
        y_val[n] = np.expand_dims(img, axis=-1)

    idx_rand = random.randint(0, len(train_img_paths))
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x_train[idx_rand])
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(y_train[idx_rand]))
    plt.axis('off')
    plt.show()

    tf.keras.backend.clear_session()
    model = model_unet_kaggle(img_height, img_width, img_channels, num_classes)
    model_fig_file = 'Output/Model_figure/segmentation_model_u_net.png'
    tf.keras.utils.plot_model(model, to_file=model_fig_file, show_shapes=True)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("FireSegmentation.h5", save_best_only=True)
    early_stopper = tf.keras.callbacks.EarlyStopping(patience=5)

    results = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopper, checkpoint])

    model_predict = tf.keras.models.load_model("FireSegmentation_fifth.h5")
    preds_val = model.predict(x_val, verbose=1)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)

    plot_segmentation_test(xval=x_val, yval=y_val, ypred=preds_val_t, num_samples=6)


def model_unet_kaggle(img_hieght, img_width, img_channel, num_classes):
    inputs = Input((img_hieght, img_width, img_channel))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
