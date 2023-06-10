import pandas as pd
import numpy as np
from PIL import Image
import os
import random

# AUGMENT_PATH = 'C:/Niranjan/Ashoka/Research/TNBC/SimilarityNet/Data/Augmented'
AUGMENT_PATH = '/storage/tnbc/gen1_label/224_aug'
# DATA_PATH = 'C:/Niranjan/Ashoka/Research/TNBC/Data/224Labelledv0.1'
DATA_PATH = '/storage/tnbc/gen1_label/224_gen1'

def create_pos_df(aug_df):
    pos_df = pd.DataFrame(index=np.arange(0, len(aug_df)),columns = ["img_1", "img_2", "similarity"])
    
    # iterate through aug df rows
    for index, row in aug_df.iterrows():
        img_1 = row['real_path']
        img_2 = row['aug_path']
        similarity = 1
        pos_df.iloc[index] = [img_1, img_2, similarity]
    
    pos_df = pos_df.dropna()
    pos_df.to_csv('./pos_df.csv')
    return pos_df

def create_neg_df():
    neg_df = pd.DataFrame(index=np.arange(0, 100000),columns = ["img_1", "img_2", "similarity"])
    
    classes = ['Tumour', 'TILs', 'Stroma', 'Fat Cells', 'White Space']
    counter = 0
    for c1 in range(len(classes)-1):
        for c2 in range(c1+1, len(classes)):
            c1_dir = os.path.join(DATA_PATH, classes[c1])
            c2_dir = os.path.join(DATA_PATH, classes[c2])
            
            c1_imgs = os.listdir(c1_dir)
            c2_imgs = os.listdir(c2_dir)
            
            for c1_img in c1_imgs:
                c2_choices = random.sample(c2_imgs, 2)
                for c2_img in c2_choices:
                    img_1 = os.path.join(c1_dir, c1_img)
                    img_2 = os.path.join(c2_dir, c2_img)
                    similarity = 0
                    neg_df.iloc[counter] = [img_1, img_2, similarity]
                    counter += 1    
        
    neg_df = neg_df.dropna()
    neg_df.to_csv('./neg_df.csv')
    return neg_df

def dataset_df_main():
    aug_df = pd.read_csv(os.path.join(AUGMENT_PATH, 'augmented.csv'), index_col=0)
    
    pos_df = create_pos_df(aug_df)
    neg_df = create_neg_df()
    
    data_df = pd.concat([pos_df, neg_df])

    data_df.to_csv('/home/niranjan.rajesh_ug23/TNBC/SimilarityNet/data_df.csv')
    return

if __name__ == "__main__":
    dataset_df_main()
