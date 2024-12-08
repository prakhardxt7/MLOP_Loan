import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.ensamble import RandomForestClassifier
import numpy as np
import pandas as pd



def eval(p1, p2):
    output_metric = (p1 ** 2) + (p2 ** 2)
    return output_metric
def load_data():
    df=pd.read_csv(URL,sep=';')

def main(alpha,l1_ratio):
    df = load_data()

if __name__ == '__main__':
    # Parse command-line arguments
    args = argparse.ArgumentParser()
    args.add_argument('--alpha', '-a', type=float, default=0.2)
    args.add_argument('--l1_ratio', '-l1', type=float, default=0.3)
    parsed_args = args.parse_args()

    # Call main function and capture the metric
    metric = main(parsed_args.param1, parsed_args.param2)

    # Print the metric (optional)
    print(f"Logged Metric: {metric}")
