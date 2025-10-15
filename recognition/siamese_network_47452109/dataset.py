"""
File must contain the data loader for loading and preprocessing your data
"""
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class MelanomaDataset(Dataset):
    """Converts jpg data into data usable by the model"""
    def __init__(self):
        super().__init__()

    def get_metadata(self, subset = True):
        """
        Reads metadata file.             \n
        :subset: True - image_name, patient_id and target columns <i>only</i>. False - all columns in metadata file.
        """
        train_groundtruth = pd.read_csv("recognition\\siamese_network_47452109\\data\\ISIC_2020_Training_GroundTruth_v2.csv")
        if subset == True: 
            return train_groundtruth[["image_name", "patient_id", "target"]]
        return train_groundtruth
    
    def get_subset(self, class_size: int, seed: int):
        """
        Gets a random subset of the data with equal amounts of benign/malicious
        """
        meta = self.get_metadata()
        benign = meta[meta["target"]== 0] 
        malicious = meta[meta["target"] == 1]

        n0 = min(class_size, len(benign))
        n1 = min(class_size, len(malicious))
        if n0 < class_size or n1 < class_size:
            print(f"Requested class_size={class_size}, but we only have "
                f"{len(benign)} benign and {len(malicious)} malicious cases. Will cap to n0={n0}, n1={n1}. :)")

        benign_sample = benign.sample(n=n0, random_state=seed)
        malicious_sample = malicious.sample(n=n1, random_state=seed)

        subset_df = pd.concat([benign_sample, malicious_sample], axis=0).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
        subset_df["file_name"] = subset_df["image_name"].astype(str) + ".jpg"
    
        return subset_df

    def preprocess_data(self):
        return 
    
