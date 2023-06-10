import os
import numpy as np
import cv2 as cv
import pandas as pd
from PIL import Image

# data_dir = 'C:/Niranjan/Ashoka/Research/TNBC/Data/224Labelledv0.1'
data_dir = '/storage/tnbc/gen1_label/224_gen1'
# output_dir = 'C:/Niranjan/Ashoka/Research/TNBC/SimilarityNet/Data/Augmented'
output_dir = '/storage/tnbc/gen1_label/224_aug'



def rotate_image(img_path, df, count):
    # load tif file and roate it 90 degrees
    img_name = os.path.basename(img_path)
    degrees = [90, 180, 270]
    rand_degree = np.random.choice(degrees)
    print(f'Rotating {img_path}')
    img = Image.open(img_path)
    img = img.rotate(rand_degree)
    new_path = os.path.join(output_dir, f'r{rand_degree}_{img_name}')
    img.save(new_path)
    df.iloc[count] = [img_name, img_path, new_path]
    return

def flip_image(img_path, df, count):
    # load tif file and roate it 90 degrees
    img_name = os.path.basename(img_path)
    flips = ['LR', 'TB']
    rand_flip = np.random.choice(flips)
    print(f'Flipping {img_path}')
    img = Image.open(img_path)
    if rand_flip == 'LR': img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif rand_flip == 'TB': img = img.transpose(Image.FLIP_TOP_BOTTOM)
    new_path = os.path.join(output_dir, f'f{rand_flip}_{img_name}')
    img.save(new_path)
    df.iloc[count] = [img_name, img_path, new_path]
    return

def augment_main():
    aug_dir_path = output_dir+'/augmented.csv'
    if os.path.exists(path=aug_dir_path): os.remove(aug_dir_path)
    positive_classes = ['Tumour', 'TILs', 'Stroma', 'Fat Cells', 'White Space']
    
    df = pd.DataFrame(index=np.arange(0, 10000),columns = ["img_name", "real_path", "aug_path"])
    
    for class_dir in positive_classes:
        class_dir_path = os.path.join(data_dir, class_dir)
        count = 0
        
        for img_name in os.listdir(class_dir_path):
            img_path = os.path.join(class_dir_path, img_name)
            
            if count % 2 == 0: rotate_image(img_path, df, count)
            else: flip_image(img_path, df, count)
            count += 1
    # save df as csv
    df = df.dropna()
    df.to_csv(aug_dir_path)
    return
        
if __name__ == "__main__":
    augment_main()