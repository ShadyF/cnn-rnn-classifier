import os

from keras.applications.xception import Xception, decode_predictions, preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.engine import Input, merge
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Lambda, K, LSTM
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

import time
import numpy as np
from keras.utils import np_utils

np.random.seed(1337)


def load_val_dataset():
    validation_data_dir = './tiny-imagenet-100-A/val/'

    classes = []
    for subdir in sorted(os.listdir(validation_data_dir)):
        if os.path.isdir(os.path.join(validation_data_dir, subdir)):
            classes.append(subdir)

    class_indices = dict(zip(classes, range(len(classes))))

    X_val = []

    # Extracting validation dat
    i = 0
    y_val = []
    for subdir in classes:
        subpath = os.path.join(validation_data_dir, subdir)
        for fname in sorted(os.listdir(subpath)):
            y_val.append(class_indices[subdir])

            # Load image as numpy array and append it to X_val
            img = load_img(os.path.join(subpath, fname), target_size=(img_width, img_height))
            x = img_to_array(img)
            X_val.append(x)

            i += 1

    Y_val = np_utils.to_categorical(y_val)
    X_val = np.asarray(X_val, dtype='float32')
    return classes, X_val, Y_val


def rgb_to_grayscale(input):
    """Average out each pixel across its 3 RGB layers resulting in a grayscale image"""
    return K.mean(input, axis=3)


def rgb_to_grayscale_output_shape(input_shape):
    return input_shape[:-1]


nb_val_samples = 5000

img_width = 299
img_height = 299

print("Building model...")
input_tensor = Input(shape=(img_width, img_height, 3))

# Creating CNN
cnn_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)

x = cnn_model.output
cnn_bottleneck = GlobalAveragePooling2D()(x)

# Creating RNN
x = Lambda(rgb_to_grayscale, rgb_to_grayscale_output_shape)(input_tensor)
x = Reshape((23, 3887))(x)  # 23 timesteps, input dim of each timestep 3887
x = LSTM(2048, return_sequences=True)(x)
rnn_output = LSTM(2048)(x)

# Merging both cnn bottleneck and rnn's output wise element wise multiplication
x = merge([cnn_bottleneck, rnn_output], mode='mul')
predictions = Dense(100, activation='softmax')(x)

model = Model(input=input_tensor, output=predictions)

model.load_weights("./finetuned_cnn_rnn_weights_2.hdf5")

print("Model built")

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

classes, X_val, Y_val = load_val_dataset()

n_labels = len(classes)

n_imgs_by_label = np.zeros(n_labels, dtype=np.dtype(int))
n_top1_accurate_by_label = np.zeros(n_labels, dtype=np.dtype(int))
n_top5_accurate_by_label = np.zeros(n_labels, dtype=np.dtype(int))

# Loop over each validation image and calculate Top-1 and Top-5 Correct Classification Rate
for i, img in enumerate(X_val):
    print(i)
    ground_truth = Y_val[i].argmax()
    n_imgs_by_label[ground_truth] += 1

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = model.predict(img)

    top_5_indices = (-preds).argsort()[:, :5]
    top_5_indices = top_5_indices[0]
    if ground_truth == top_5_indices[0]:
        n_top1_accurate_by_label[ground_truth] += 1
    if ground_truth in top_5_indices:
        n_top5_accurate_by_label[ground_truth] += 1

# Create a text file that contains the top 1 and top 5 ACCR of each label
results_path = '/home/shady-fanous/cnn_rnn_results.txt'
with open(results_path, 'w+') as f:
    f.write('Label\tTop-1 Accuracy\tTop-5 Accuracy\n')
    for i, label in enumerate(classes):
        label_top1_accuracy = round(100.0 * n_top1_accurate_by_label[i] / n_imgs_by_label[i], 2)
        label_top5_accuracy = round(100.0 * n_top5_accurate_by_label[i] / n_imgs_by_label[i], 2)
        line = '{}\t{}\t{}\n'.format(label, label_top1_accuracy, label_top5_accuracy)
        f.write(line)
