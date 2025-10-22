"""
File must contain the data loader for loading and preprocessing your data
"""
import os
import global_vars as gv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import random
from sklearn.model_selection import train_test_split

# Helper functions: 
def open_rgb(path):
    img = Image.open(path)
    return img.convert("RGB") if img.mode != "RGB" else img

class MelanomaDataset(Dataset):
    """
    Reads raw data and gets it ready for machine learning
    """
    def __init__(self, root, allow_transforms = True):
        self.root = root # folder with data
        self.metadata = self._read_metadata(root) 
        self.allow_transforms = allow_transforms

        self.sort = {0: [], 1: []}
        for i, row in self.metadata.iterrows():
            target = int(row["target"])
            self.sort[target].append(i)

        super().__init__()

    def __getitem__(self, index):
        data = self.metadata.iloc[index]
        # Get image 1 at index
        img_name = data["image_name"]
        img_path = os.path.join(self.root, "train", img_name + ".jpg")
        img1 = open_rgb(img_path)

        target1 = data["target"]
        
        # Get random image 2
        alt_label = random.choice([0,1])

        img2_idx = random.choice(self.sort.get(alt_label))
        img2_name = self.metadata.iloc[img2_idx]["image_name"]
        img2_path = os.path.join(self.root, "train", img2_name + ".jpg")
        img2 = open_rgb(img2_path)

        target2 = self.metadata.iloc[img2_idx]["target"]

        if self.allow_transforms:
            # 50% chance of transforming images
            if random.random() > 0.5:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

        if target1 == target2:
            label = 0
        else: label = 1

        return img1, img2, torch.tensor(label)
    
    def __len__(self):
        return len(self.metadata)
    
    def _read_metadata(self, root: str) -> pd.DataFrame:
        """
        Reads metadata file.             

        :param images_dir: Image directory
        """
        train_groundtruth = pd.read_csv(os.path.join(root, "ISIC_2020_Training_GroundTruth_v2.csv"))
        self.metadata = train_groundtruth
        return train_groundtruth
    
    def set_subset(self, class_size: int, seed: int):
        """
        Sets data to a random subset of the data with equal amounts of benign/malignant targets

        :param class_size: Number of samples from each class. Subset size will be 2*class_size.
        :returns: dataframe with subset of data with equal amounts of each class as per class_size

        TODO: REMOVE ME BEFORE SUBMISSION. FOR TESTING PURPOSES ONLY.
        """
        meta = self._read_metadata(self.root)
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

        return subset_df
   
    def transform(self, img):
        """transform function"""
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return trans(img)

    def split_data(self, data: pd.DataFrame, seed):
        """
        Split into Train, Validate and Test datasets
        splits into 0.75:0.15:0.1
        """
        train, spare = train_test_split(
            data,
            test_size= 0.25,
            stratify=data["target"],
            random_state = seed
        )

        validate, test = train_test_split(
            spare,
            test_size= 0.4,
            stratify=spare["target"],
            random_state = seed
        )

        return train, validate, test