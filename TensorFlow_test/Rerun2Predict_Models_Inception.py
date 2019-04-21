#https://www.tensorflow.org/tutorials/keras/save_and_restore_models 
# Recreate the exact same model, including weights and optimizer.
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import os

# model directory
saved_model_path = r'F:\\CBIS_DDSM_PNG\\Feature_Keras_inception_v3\\1555720707_Masked_EPOCH100_2'
# predict images directory
predict_data_root = (r'F:\\CBIS_DDSM_PNG\\MASKED\\Calc_Mask_v0_3')

# Generate Data from directory
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

# [Classifier]
feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"

def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

# [RESTORE model]
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
#new_model.summary()

# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.

new_model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# [TFHub Initialize] - based on current parameters
import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

#  PREDICTIONS
# Check Prediction Suffle false to not reorder in index
image_dataPredict_numfile = image_generator.flow_from_directory(str(predict_data_root),shuffle=False, target_size=IMAGE_SIZE)

nb_samples = len(image_dataPredict_numfile.filenames)

image_dataPredict = image_generator.flow_from_directory(str(predict_data_root),shuffle=False, target_size=IMAGE_SIZE ,batch_size = nb_samples) # change batch size check why?
for image_batch,label_batch in image_dataPredict:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break

#FInd filenames

result_batch = new_model.predict_generator(image_dataPredict, steps = nb_samples)

label_names = sorted(image_dataPredict.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])
labels_batch = label_names[np.argmax(result_batch, axis=-1)]

#print (result_batch)

#https://stackoverflow.com/questions/49973379/how-to-get-associated-image-name-when-using-predict-generator-in-keras

label_filenames = np.array(image_dataPredict.filenames)
acc = 0
prediction_acc = ""
with open(os.path.join(predict_data_root,'Prediction.csv'), mode='w') as prediction_file:

  for n in range(nb_samples):
    #print (label_filenames[n])
    label_folder, label_filename = label_filenames[n].split("\\")
    if label_folder.lower() == labels_batch[n].lower(): #ignore case
      acc += 1
    prediction_file.write(label_filename+","+label_folder+","+labels_batch[n]+","+ str(result_batch[n][0])+","+str(result_batch[n][1])+"\n")
  
  prediction_file.write(str(acc/nb_samples))
  prediction_acc= str(acc/nb_samples * 100)
  print (prediction_acc)
'''
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