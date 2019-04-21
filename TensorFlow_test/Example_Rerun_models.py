#https://www.tensorflow.org/tutorials/keras/save_and_restore_models 
# Recreate the exact same model, including weights and optimizer.
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
# [DATA INPUT] Where You put your Data
data_root = (r'D:\\CBIS_DDSM_PNG\\unmasked_raw')

# Generate Data from directory
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root))

# [Classifier]
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

# Ensure the data generator is generating images of the expected size
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break

# [RESTORE model]
saved_model_path = 'D:\\CBIS_DDSM_PNG\\1552308961-keras\\'
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model.summary()

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


# Check Prediction
label_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])
print(label_names)

result_batch = new_model.predict(image_batch)
print(result_batch)

labels_batch = label_names[np.argmax(result_batch, axis=-1)]
print(labels_batch)

#score, acc = new_model.evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)

#print("Restored model, accuracy: {:5.2f}%".format(100*acc))
quit()

