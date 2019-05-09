'''
Rerun2Predict_v0_1.py
[Goal]
1. Re-compile model
2. Load in test data set
3. Calculate Accuracy
https://www.tensorflow.org/tutorials/keras/save_and_restore_models 
'''
#https://www.tensorflow.org/tutorials/keras/save_and_restore_models 
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import os

# Start time to test how long it runs
import time

# Model directory
saved_model_path = r''
# prediction directory
predict_data_root = r''
# Classifier
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

# SET UP model that has frozen layer again!
# Create the module:
def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)
# Wrap module in Keras layer
features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])
# Freeze the variables in the feature extractor layer, so that the training only modifies the new classifier layer.
features_extractor_layer.trainable = False

# Generate Data from directory
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
# Load image from directory, shuffle=false to disable indexing, to allow ease of listing later
image_Predict = image_generator.flow_from_directory(str(predict_data_root),shuffle=False, target_size=IMAGE_SIZE,class_mode='binary')
# Obtain number of files
nb_samples = len(image_Predict.filenames)
# RESTORE model 
restore_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
restore_model.summary()
# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.
restore_model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# [TFHub Initialize] - based on current parameters
import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

# Find filenames
result_batch = restore_model.predict_generator(image_Predict, steps = nb_samples)
label_names = sorted(image_Predict.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])
labels_batch = label_names[np.argmax(result_batch, axis=-1)]

# https://stackoverflow.com/questions/49973379/how-to-get-associated-image-name-when-using-predict-generator-in-keras
label_filenames = np.array(image_Predict.filenames)
acc = 0
prediction_acc = ""
# Create Folder with time stamp of completion
st = time.strftime('%H-%M%p-%d-%b-%Y')
with open(os.path.join(saved_model_path,'Prediction'+st+'.csv'), mode='w') as prediction_file:

  for n in range(nb_samples):
    label_folder, label_filename = label_filenames[n].split("\\")
    if label_folder.lower() == labels_batch[n].lower(): #ignore case
      acc += 1
    prediction_file.write(label_filename+","+label_folder+","+labels_batch[n]+","+ str(result_batch[n][0])+","+str(result_batch[n][1])+"\n")
  
  prediction_file.write(str(acc/nb_samples))
  prediction_acc= str(acc/nb_samples * 100)
  print (prediction_acc)
'''
# find label batches
for image_batch,label_batch in image_Predict:
  break

plt.figure(figsize=(10,9))

for n in range(nb_samples):

  plt.subplot(22,13,n+1)
  plt.imshow(image_batch[n])
  label_folder, label_filename = label_filenames[n].split("\\")
  plt.title(label_folder+"_"+label_filename.replace(".png","")+"_"+labels_batch[n], fontsize=5)
  plt.axis('off')
_ = plt.suptitle("Model predictions: "+prediction_acc)

plt.show()
'''