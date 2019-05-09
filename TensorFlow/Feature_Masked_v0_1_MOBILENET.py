'''
Reference = https://www.tensorflow.org/tutorials/images/hub_with_keras
[Goal]
1. Load Data set
2. Load network
3. Prepare data set for network
4. Fit data set into network
5. Plot progress and show time taken
6. Save network
'''
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import os  

# Start time to test how long it runs
import time
start_time = time.time()

# [DATA INPUT] Where You put your Data
data_root = (r'F:\\CBIS_DDSM_PNG\\MASKED\\Calc_Mask_v0_3')
export_path = (r"F:\\CBIS_DDSM_PNG\\Feature_Keras_mobilenet_v2_100_224")

# Generate Data from directory
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root))

# Network Vector selected:
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

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

# Training Parameters
steps_per_epoch = image_data.samples//image_data.batch_size
batch_stats = CollectBatchStats()
epochSelected = 1

model.fit((item for item in image_data), epochs=epochSelected, 
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats])
                    #,workers=4,use_multiprocessing=True)
                    
timetaken = ("--- %s seconds ---" % (time.time() - start_time))
print(timetaken)

# Create Folder with time stamp of completion
st = time.strftime('%H-%M%p-%d-%b-%Y')
folderOut = os.path.join(export_path,st)
os.makedirs(folderOut)

# Store training details
file = open(os.path.join(folderOut,"info.txt"), "w") 
file.write("time taken:"+timetaken +"\n"+"feature_extractor_url:"+feature_extractor_url+"\n"+"Epoch:"+str(epochSelected)+"\n"+"data_root"+data_root) 
file.close() 

# Plt progress
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats.batch_losses)
plt.savefig(os.path.join(folderOut,'lossVStraining.png'))

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats.batch_acc)
plt.savefig(os.path.join(folderOut,'accVStraining.png'))

#plt.show()
export_path = tf.contrib.saved_model.save_keras_model(model, folderOut)
print(export_path)