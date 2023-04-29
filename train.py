import tensorflow as tf
import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from model import get_model


def make_tf_dataset():
    
    return

def main():
    data_df = pd.read_csv('./data_df.csv', index_col=0)
    model = get_model()
    
    make_tf_dataset(data_df)
    print(model.summary())
    
if __name__ == "__main__":
    main()