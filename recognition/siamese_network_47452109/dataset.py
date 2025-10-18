"""
File must contain the data loader for loading and preprocessing your data
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import random

#TODO: Augmentation?
#TODO: Transform

SEED = 42

# Helper functions: 
def open_rgb(path):
    img = Image.open(path)
    return img.convert("RGB") if img.mode != "RGB" else img

class MelanomaDataset(Dataset):
    """
    Reads raw data and gets it ready for machine learning
    """
    def __init__(self, root):
        self.root = root # folder with data
        self.metadata = None 
        self.subset_df = None
        super().__init__()

    def __getitem__(self, index):
        return
    
    def _read_metadata(self, images_dir: str, subset = False) -> pd.DataFrame:
        """
        Reads metadata file.             

        :param images_dir: Image directory
        :param subset: True - image_name, patient_id and target columns <i>only</i>. False - all columns in metadata file.
        :returns: metadata file as dataframe
        """
        self.root = images_dir
        train_groundtruth = pd.read_csv(self.root + "ISIC_2020_Training_GroundTruth_v2.csv")
        
        if subset: 
            self.metadata = train_groundtruth
            return train_groundtruth[["image_name", "patient_id", "target"]]
        
        self.metadata = train_groundtruth
        return train_groundtruth
    
    def get_subset(self, class_size: int, seed: int, reduced = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Gets a random subset of the data with equal amounts of benign/malignant

        :param class_size: Number of samples from each class. Subset size will be 2*class_size.
        :returns: dataframe with subset of data with equal amounts of each class as per class_size
        """
        meta = self._read_metadata(self.root, reduced)
        benign = meta[meta["target"]== 0] 
        malignant = meta[meta["target"] == 1]

        n0 = min(class_size, len(benign))
        n1 = min(class_size, len(malignant))
        if n0 < class_size or n1 < class_size:
            print(f"Requested class_size={class_size}, but we only have "
                f"{len(benign)} benign and {len(malignant)} malignant cases. Will cap to n0={n0}, n1={n1}. :)")

        benign_sample = benign.sample(n=n0, random_state=seed)
        malignant_sample = malignant.sample(n=n1, random_state=seed)

        subset_df = pd.concat([benign_sample, malignant_sample], axis=0)\
            .sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
        subset_df["file_name"] = subset_df["image_name"].astype(str) + ".jpg"

        self.subset_df = subset_df

        return subset_df, benign_sample, malignant_sample

    def transform(self, x):
        """will be updated to transform"""
        return x

    def pair_generator(self, data: pd.DataFrame, proportion_pos = 0.5, seed = SEED):
        """
        pair generator.
        """
        img_dir = self.root + "train\\"
        sort = {0: [], 1: []}
        for i, row in data.iterrows():
            target = int(row["target"])
            sort[target].append(i)

        rng = random.Random(seed)

        while True:
            
            same = rng.random() < proportion_pos

            if same: # two of same class
                choice = rng.choice([0,1])
                a,b = rng.sample(sort[choice], 2)
                label = 1

            else: # different classes
                a = rng.choice(sort[0])
                b = rng.choice(sort[1])
                label = 0

            r1 = data.iloc[a]
            r2 = data.iloc[b]
            p1 = img_dir + r1["file_name"]
            p2 = img_dir + r2["file_name"]

            img1 = open_rgb(p1)
            img2 = open_rgb(p2)

            img1 = self.transform(img1)
            img2 = self.transform(img2)
            yield img1, img2, label


# Test
path = "recognition\\siamese_network_47452109\\data\\"

md = MelanomaDataset(path)
sub, ben, mal = md.get_subset(50, SEED)

print(md.pair_generator(sub, 1.0))

