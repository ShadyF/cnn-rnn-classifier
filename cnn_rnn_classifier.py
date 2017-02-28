from keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint
from keras.engine import Input, merge
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Lambda, K, LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

import time
import numpy as np

np.random.seed(1337)


class CustomImageDataGenerator(ImageDataGenerator):
    """
    Because Xception utilizes a custom preprocessing method, the only way to utilize this
    preprocessing method using the ImageDataGenerator is to overload the standardize method.

    The standardize method gets applied to each batch before ImageDataGenerator yields that batch.
    """

    def standardize(self, x):
        """
        Taken from keras.applications.xception.preprocess_input
        """
        if self.featurewise_center:
            x /= 255.
            x -= 0.5
            x *= 2.
        return x


def get_training_generator(batch_size=128):
    train_data_dir = './tiny-imagenet-100-A/train/'
    validation_data_dir = './tiny-imagenet-100-A/val/'
    image_datagen = CustomImageDataGenerator(featurewise_center=True)

    train_generator = image_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size
    )

    val_generator = image_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, val_generator


def rgb_to_grayscale(input):
    """Average out each pixel across its 3 RGB layers resulting in a grayscale image"""
    return K.mean(input, axis=3)


def rgb_to_grayscale_output_shape(input_shape):
    return input_shape[:-1]


batch_size_phase_one = 32
batch_size_phase_two = 16
nb_val_samples = 5000

nb_epochs = 30

img_width = 299
img_height = 299

# Setting tensorbord callback
now = time.strftime("%c")
tensorboard_callback = TensorBoard(log_dir='./logs/' + 'cnn_rnn ' + now, histogram_freq=0, write_graph=True,
                                   write_images=False)

# Loading dataset
print("Loading the dataset with batch size of {}...".format(batch_size_phase_one))
train_generator, val_generator = get_training_generator(batch_size_phase_one)
print("Dataset loaded")

print("Building model...")
input_tensor = Input(shape=(img_width, img_height, 3))

# Creating CNN
cnn_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)

x = cnn_model.output
cnn_bottleneck = GlobalAveragePooling2D()(x)

# Make CNN layers not trainable
for layer in cnn_model.layers:
    layer.trainable = False

# Creating RNN
x = Lambda(rgb_to_grayscale, rgb_to_grayscale_output_shape)(input_tensor)
x = Reshape((23, 3887))(x)  # 23 timesteps, input dim of each timestep 3887
x = LSTM(2048, return_sequences=True)(x)
rnn_output = LSTM(2048)(x)

# Merging both cnn bottleneck and rnn's output wise element wise multiplication
x = merge([cnn_bottleneck, rnn_output], mode='mul')
predictions = Dense(100, activation='softmax')(x)

model = Model(input=input_tensor, output=predictions)

print("Model built")

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

print("Starting training")
checkpointer = ModelCheckpoint(filepath="./initial_cnn_rnn_weights_2.hdf5", verbose=1, save_best_only=True)
model.fit_generator(train_generator, samples_per_epoch=4480, nb_epoch=nb_epochs, verbose=1,
                    validation_data=val_generator,
                    nb_val_samples=nb_val_samples,
                    callbacks=[tensorboard_callback, checkpointer])

print("Initial training done, starting phase two (finetuning)")

# Load two new generator with smaller batch size, needed because using the same batch size
# for the fine tuning will result in GPU running out of memory and tensorflow raising an error
print("Loading the dataset with batch size of {}...".format(batch_size_phase_two))
train_generator, val_generator = get_training_generator(batch_size_phase_two)
print("Dataset loaded")

# Load best weights from initial training
model.load_weights("./initial_cnn_rnn_weights_2.hdf5")

# Make all layers trainable for finetuning
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

checkpointer = ModelCheckpoint(filepath="./finetuned_cnn_rnn_weights_2.hdf5", verbose=1, save_best_only=True,
                               monitor='val_acc')
model.fit_generator(train_generator, samples_per_epoch=2240, nb_epoch=nb_epochs, verbose=1,
                    validation_data=val_generator,
                    nb_val_samples=nb_val_samples,
                    callbacks=[tensorboard_callback, checkpointer])

# Final evaluation of the model
print("Training done, doing final evaluation...")

model.load_weights("./finetuned_cnn_rnn_weights_2.hdf5")

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

scores = model.evaluate_generator(val_generator, val_samples=nb_val_samples)
print(model.metrics_names, scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
