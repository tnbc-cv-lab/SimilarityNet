import tensorflow as tf
import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from model import get_model
from utils import plot_accuracy


def make_tf_dataset(df):

    # Convert the image paths to numpy arrays
    img1 = df["img_1"].values
    img2 = df["img_2"].values

    # Load the images as numpy arrays
    img1 = [tf.keras.preprocessing.image.load_img(path, target_size=(224, 224)) for path in img1]
    img2 = [tf.keras.preprocessing.image.load_img(path, target_size=(224, 224)) for path in img2]

    # Convert the images to numpy arrays and normalize them
    img1 = np.array([tf.keras.preprocessing.image.img_to_array(img) / 255.0 for img in img1])
    img2 = np.array([tf.keras.preprocessing.image.img_to_array(img) / 255.0 for img in img2])

    # Convert the similarity scores to numpy arrays
    similarity = df["similarity"].values

    # Create a TensorFlow dataset from the numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((img1, img2, similarity))
    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.map(lambda img1, img2, similarity: ((img1, img2), similarity))

    train_size = int(len(df) * 0.8)
    valid_size = len(df) - train_size

    train_dataset = dataset.take(train_size)
    valid_dataset = dataset.skip(train_size).take(valid_size)
    
    return train_dataset, valid_dataset

def train_main():
    data_df = pd.read_csv('/home/niranjan.rajesh_ug23/TNBC/SimilarityNet/data_df.csv', index_col=0)
    model = get_model()
    
    train_ds, valid_ds = make_tf_dataset(data_df)
    
    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    history = model.fit(train_ds, epochs=20, validation_data=valid_ds, batch_size=32)
    plot_accuracy(history)
    
    
if __name__ == "__main__":
    train_main()