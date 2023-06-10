import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
new_model = load_model('C:/Niranjan/Ashoka/Research/TNBC/SimilarityNet/Models/SimNet_0.h5')
print(new_model.summary())
path1 = "C:/Niranjan/Ashoka/Research/TNBC/Data/224Labelledv0.1/White Space/20190610_100_401-15_4350-15-2_Biopsy_ER_HnE_40X_cropped_10805_58209_12671.tif"
path2 = "C:/Niranjan/Ashoka/Research/TNBC/Data/224Labelledv0.1/White Space/20190610_100_401-15_4350-15-2_Biopsy_ER_HnE_40X_cropped_10805_58209_12671.tif"

img1 = tf.keras.preprocessing.image.load_img(path1, target_size=(224, 224))
img2 = tf.keras.preprocessing.image.load_img(path2, target_size=(224, 224))

img1 = np.array(tf.keras.preprocessing.image.img_to_array(img1) / 255.0)
img2 = np.array(tf.keras.preprocessing.image.img_to_array(img2) / 255.0)

img1 = np.expand_dims(img1, axis=0)
img2 = np.expand_dims(img2, axis=0)

out = new_model.predict([img1, img2])
print(out)