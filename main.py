import os
from augments import augment_main
from dataset import dataset_df_main
from train import train_main

def make_dataset():
    augment_main()
    dataset_df_main()

if __name__ == "__main__":
    get_dataset()
    train_main()