'''
Reference = https://www.tensorflow.org/tutorials/images/hub_with_keras
'''
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#quit()

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np  

# [DATA INPUT] Where You put your Data
data_root = (r'D:\\CBIS_DDSM_PNG\\UNMASKED\\CALC')

# Generate Data from directory
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root))

# Network Vector selected:
feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"


# Create the module, and check the expected image size:
def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))


# "AUTO DOWNSAMPLING" Ensure the data generator is generating images of the expected size 
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break

# Wrap module in Keras layer
features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])

# Freeze the variables in the feature extractor layer, so that the training only modifies the new classifier layer.
features_extractor_layer.trainable = False

# [Attach classification head]
model = tf.keras.Sequential([
  features_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])
model.summary()

# [TFHub Initialize] - based on current parameters
import tensorflow.keras.backend as K

sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
quit()
# Test single bach
#result = model.predict(image_batch)
#print (result.shape)
#quit()

# [Train model]
model.compile(
  optimizer=tf.train.AdamOptimizer(), 
  loss='categorical_crossentropy',
  metrics=['accuracy'])
  
#Now use the .fit method to train the model.
#To keep this example short train just a single epoch. To visualize the training progress during that epoch, use a custom callback to log the loss and accuract of each batch.
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])

steps_per_epoch = image_data.samples//image_data.batch_size

batch_stats = CollectBatchStats()

model.fit((item for item in image_data), epochs=100, 
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats])

# Plt progress
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats.batch_acc)

# plt.show()
export_path = tf.contrib.saved_model.save_keras_model(model, r"D:\\CBIS_DDSM_PNG\\Classification_Keras_inception_v3")
print(export_path)

quit()

# PREDICTIONS
# Check Prediction # modify this!
image_dataPredict = image_generator.flow_from_directory(str(data_root),shuffle=False, target_size=IMAGE_SIZE)

#FInd filenames
#print (image_data.filenames)
label_filenames = np.array(image_dataPredict.filenames)

nb_samples = len(image_dataPredict.filenames)
result_batch = model.predict_generator(image_dataPredict, steps = nb_samples)

class_names = np.array(result_batch.argmax(axis=-1))

label_names = sorted(image_dataPredict.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])
labels_batch = np.array(label_names[np.argmax(result_batch, axis=-1)])

prediction = np.column_stack((label_filenames,class_names,labels_batch))
np.savetxt("test.csv", prediction, delimiter=",")