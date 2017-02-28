This is a practical example on how to combine both a CNN and a RNN to classify images.

_NOTE: This classifier was tested with the tiny-imagenet-100 dataset only._

## Network Architecture

The network consists of two different branches: a CNN branch which uses the Xception model, pretrained
on imagenet and provided by Keras (https://keras.io/applications/#xception) and another indepented RNN branch.

Each one of these branches runs parallel to each other.

Initially, the entire network takes an RGB image whose shape is 299x299x3.

On the CNN branch, this image is taken as is (299x299x3) and passed through the pretrained Xception
model until it reaches the final convolution block which has the bottleneck features, which is of size
(batch_size, 2048).

On the other branch, the 299x299x3 image is transformed into a grayscale image of size 299x299x1 to
be able to properly split it into chunks to feed it into the RNN. Afterwards, this 299x299 image is
reshaped into (23, 3887), where 23 is the timesteps and 3887 is the dim of each timestep. These values
were chosen because 23*3887 == 299*299. The reshaped image is then passed through two LSTM
layers, each of which are of (batch_size, 2048) output.

Next, now that we have (batch_size, 2048) from both the CNN and RNN branches, these two outputs
are merged using element-wise multiplication. The output of this multiplication is then fed to the
classification layer which consists of 100 nodes (100 classes) and a softmax activation.

## Network Training

The network was trained in two phases. In the first phase, all the layers of the CNN were frozen and only
the last classification layer and the RNN network were trained. This was done using the RMSProp
optimizer.

In the second phase, all the layers of the entire network were unfrozen and finetuned using Adam
optimizer with a learning rate of 0.0001.

Using this two phase training technique, the cnn/rnn model combination is able to achieve a Top 5 Accuracy of 96.14% on 
a minified version of the ImageNet dataset that contains only 100 classes (tiny-imagenet-100)

## Dataset Structure

Keras’ ImageDataGenerator flow_from_directory method
expects the dataset to be in a certain structure. 

The restructure_dataset.py script in the helpers directory can be used
to reorganize the original dataset (given it has the same structure as the tiny-imagenet-100 dataset) into the strucutre Keras
expects.

## Image Preprocessing

The Xception model expects images to be processed in a certain way. However, because
Keras’ built in ImageDataGenerator is used, We could not easily preprocess the input while using the
fit_generator() training method.

Consequently, in cnn_rnn_classifier.py, a new class was created, CustomImageDataGenerator that inherits
from ImageDataGenerator and has an overloaded standardize() method which is called by
ImageDataGenerator before batch is yielded to fit_generator().

The standardize() method of CustomImageDataGenerator applies the Xception model’s required
preprocessing on the input.
