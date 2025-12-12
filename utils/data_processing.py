import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler


def process_data(dataset_name, I):
    dataset_path = f"./datasets/{dataset_name}_Biomarker_Input.csv"
    print(f"{I} Loading dataset {dataset_path}...")
    df = pd.read_csv(dataset_path)

    print(f"{I} Processing the data...")
    gene_names = df.drop(columns=['Unnamed: 0','target']).columns

    data = df.drop(columns=['Unnamed: 0','target']).values
    # CRUCIAL step, normalize the data
    data = RobustScaler().fit_transform(data)
    # Should avoid data copy according to
    # https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    data = np.ascontiguousarray(data)

    target = df['target']

    print(f"Data: {data.shape}")
    print("Labels: ")
    unique, counts = np.unique(target, return_counts=True)
    for u,c in zip(unique, counts):
        print(f"\t{u}: {c} samples")

    # label MDS = 1, Healthy = 0
    labels = LabelEncoder().fit_transform(target)

    return data, labels, gene_names
