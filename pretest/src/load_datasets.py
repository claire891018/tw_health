import pandas as pd
import random

def load_and_shuffle_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df.sample(frac=1).reset_index(drop=True)
    return df
