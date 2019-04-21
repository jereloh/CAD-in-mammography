#https://www.tensorflow.org/tutorials/keras/save_and_restore_models 

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import os

# model directory
saved_model_path = r'F:\\CBIS_DDSM_PNG\\Feature_Keras_inception_v3\\1555720707_Masked_EPOCH100_2'
# predict images directory
predict_data_root = (r'F:\\CBIS_DDSM_PNG\\MASKED\\Calc_Mask_v0_3_Testing')
# [Classifier] used during training and its input image_size
feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
# [RESTORE model]
restore_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
restore_model.summary()
# [Generate correct images for testing]
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_evaluate = image_generator.flow_from_directory(str(predict_data_root),shuffle=True, target_size=IMAGE_SIZE,class_mode='binary' )
# [COmpile model]
restore_model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
# [TFHub Initialize] - based on current parameters
import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

# Evaluation
nb_samples = len(image_evaluate.filenames)
results_evaluate  = restore_model.evaluate_generator(image_evaluate,nb_samples/32)
print('test loss, test acc:', results_evaluate)
# Prediction
image_Predict = image_generator.flow_from_directory(str(predict_data_root),shuffle=False, target_size=IMAGE_SIZE,class_mode='binary' )
result_predict = restore_model.predict_generator(image_Predict, steps = nb_samples)
correct = 0
for i, n in enumerate(image_Predict.filenames):
    if n.startswith("MALIGNANT") and result_predict[i][0] <= 0.5:
        correct += 1
    if n.startswith("BENIGN") and result_predict[i][0] > 0.5:
        correct += 1
print("Correct:", correct, " Total: ", len(image_Predict.filenames), " Acc: ", str(correct/len(image_Predict.filenames)))
