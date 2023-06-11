import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
new_model = load_model('C:/Niranjan/Ashoka/Research/TNBC/SimilarityNet/Models/Results/SimNet_3.h5')

DATA_PATH = "C:/Niranjan/Ashoka/Research/TNBC/Data/224Labelledv0.1"
RESULTS_PATH = "C:/Niranjan/Ashoka/Research/TNBC/SimilarityNet/Test_Results"
if os.path.exists(RESULTS_PATH):
    shutil.rmtree(RESULTS_PATH)
os.mkdir(RESULTS_PATH)

class_names = os.listdir(DATA_PATH)
class_names.remove('Black Space')

for i in range(len(class_names)):
    c1 = class_names[i]
    for j in range(i, len(class_names)):
        for n in range(2):
            c2 = class_names[j]
            
            c1_path = os.path.join(DATA_PATH, c1)
            c2_path = os.path.join(DATA_PATH, c2)
            
            # Get a list of TIF files in c1 and c2
            c1_files = os.listdir(c1_path)
            c2_files = os.listdir(c2_path)
            
            # choose random item in c1_files
            rand_img1 = np.random.choice(c1_files)
            rand_img2 = np.random.choice(c2_files)
            
            # 0'th comparison for same class will be with same pics
            if (i==j and n == 0):
                image1_path = os.path.join(c1_path, rand_img1)
                image2_path = os.path.join(c2_path, rand_img1)
            else:
                image1_path = os.path.join(c1_path, rand_img1)
                image2_path = os.path.join(c2_path, rand_img2)
            
            # Read the images
            image1 = plt.imread(image1_path)
            image2 = plt.imread(image2_path)
            
            # Plot the images side by side
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(image1)
            axes[0].set_title(c1)
            axes[1].imshow(image2)
            axes[1].set_title(c2)
            
            
            # Similarity Preds for each pair of images
            img1 = tf.keras.preprocessing.image.load_img(image1_path, target_size=(224, 224))
            img2 = tf.keras.preprocessing.image.load_img(image2_path, target_size=(224, 224))
            
            img1 = np.array(tf.keras.preprocessing.image.img_to_array(img1) / 255.0)
            img2 = np.array(tf.keras.preprocessing.image.img_to_array(img2) / 255.0)
            
            img1 = np.expand_dims(img1, axis=0)
            img2 = np.expand_dims(img2, axis=0)
            
            score = new_model.predict([img1, img2])[0][0]
            
            text = f'Similarity Score: {score:.4f}'
            fig.text(0.5, 0.05, text, ha='center', fontsize=14)
            
            # Adjust the layout and display the plot
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_PATH, c1 + "x" + c2 + "_" + str(n) + ".png"))
            plt.close()


# # TEST LOOP
# path1 = "C:/Niranjan/Ashoka/Research/TNBC/Data/224_gen1/TILs/20190610_52_69-15_G-15-678-A_Biopsy_ER_HnE_40X_cropped_14061_14061_19286.tif"
# path2 = "C:/Niranjan/Ashoka/Research/TNBC/Data/224_gen1/Tumour/20190610_3_201-17_G-17-1135_Biopsy_TNBC_HnE_40x_cropped_47082_16119_11403.tif"






# print(out)