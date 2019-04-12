'''
Learning Keras for image classification using imagenet
$ export TF_CPP_MIN_LOG_LEVEL=2
Reference = https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/hub_with_keras.ipynb#scrollTo=PLcqg-RmsLno

'''

from __future__ import absolute_import, division, print_function

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

# Load Data

# Download Data
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

# Generate Data from directory
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root))

# Image properties
for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break

# ImageNet Classifier
classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" #@param {type:"string"}
def classifier(x):
  classifier_module = hub.Module(classifier_url)
  return classifier_module(x)
  
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))

classifier_layer = layers.Lambda(classifier, input_shape = IMAGE_SIZE+[3])
classifier_model = tf.keras.Sequential([classifier_layer])
classifier_model.summary()

# Rebuild data generator, output size to match what's expected by module
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break


# Manually start TFHub modules
import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()

sess.run(init)

# Run on single image
import numpy as np
import PIL.Image as Image

#Download a single image to try the model on.
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SIZE)
#grace_hopper.show()

grace_hopper = np.array(grace_hopper)/255.0
print (grace_hopper.shape)

# Add a batch dimension, and pass the image to the model.
result = classifier_model.predict(grace_hopper[np.newaxis, ...])
print (result.shape)


# The result is a 1001 element vector of logits, rating the probability of each class for the image.
# So the top class ID can be found with argmax:

predicted_class = np.argmax(result[0], axis=-1)
print (predicted_class)

# Decode Predictions
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name)
# plt.show()

# Run on batch images
result_batch = classifier_model.predict(image_batch)

labels_batch = imagenet_labels[np.argmax(result_batch, axis=-1)]
print(labels_batch)

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(labels_batch[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")

plt.show()