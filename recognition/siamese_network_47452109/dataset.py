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
    def __init__(self, root, allow_transforms = True, oversample_ratio=1.0,is_test=False):
        self.root = root # folder with data
        self.metadata = self._read_metadata(root)
        self.allow_transforms = allow_transforms
        self.oversample_ratio = oversample_ratio
        self.is_test = is_test

        # sort data based on target value
        self.sort = {0: [], 1: []}
        for i, row in self.metadata.iterrows():
            target = int(row["target"])
            self.sort[target].append(i)

         # Calculate weights for oversampling (training ONLY)
         # weights help to balance target types
        if not self.is_test:
            targets = self.metadata['target'].values
            class_counts = self.metadata['target'].value_counts()
            num_samples = len(self.metadata)
            class_weights = {i: num_samples / count for i, count in class_counts.items()}
            self.weights = [class_weights[target] for target in targets]


    def __getitem__(self, index):
        # Open image 1 at index
        row1 = self.metadata.iloc[index]
        img_name1 = row1["image_name"]
        img_path1 = os.path.join(self.root, "train", img_name1 + ".jpg")
        img1 = open_rgb(img_path1)
        target1 = int(row1["target"])

        if self.is_test:
            # For testing, return pairs and a binary label
            make_same_class = random.random() < 0.5 # Randomly create same or different class pairs for testing
            if make_same_class:
                target2 = target1
                # Select a random index from the same class, making sure it's not the anchor itself
                idx2 = random.choice([i for i in self.sort[target2] if i != index])
            else:
                target2 = 1 - target1
                idx2 = random.choice(self.sort[target2])

            # Open image 2
            row2 = self.metadata.loc[idx2]
            img_name2 = row2["image_name"]
            img_path2 = os.path.join(self.root, "train", img_name2 + ".jpg")
            img2 = open_rgb(img_path2)

            label = 1 if target1 == target2 else 0

            # Transform images
            if self.allow_transforms:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            else: # Apply only necessary transforms for evaluation
                 eval_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                 img1 = eval_transforms(img1)
                 img2 = eval_transforms(img2)


            return img1, img2, torch.tensor(label, dtype=torch.long)

        else:
            # For training/validation, return triplets
            anchor_row = self.metadata.iloc[index]
            anchor_img_name = anchor_row["image_name"]
            anchor_img_path = os.path.join(self.root, "train", anchor_img_name + ".jpg")
            anchor_img = open_rgb(anchor_img_path)
            anchor_target = int(anchor_row["target"])

            # Get positive image (same class as anchor)
            positive_target = anchor_target
            # Select a random index from the same class, making sure it's not the anchor itself
            positive_idx = random.choice([i for i in self.sort[positive_target] if i != index])
            positive_img_name = self.metadata.loc[positive_idx]["image_name"]
            positive_img_path = os.path.join(self.root, "train", positive_img_name + ".jpg")
            positive_img = open_rgb(positive_img_path)

            # Get negative image (different class than anchor)
            negative_target = 1 - anchor_target
            negative_idx = random.choice(self.sort[negative_target])
            negative_img_name = self.metadata.loc[negative_idx]["image_name"]
            negative_img_path = os.path.join(self.root, "train", negative_img_name + ".jpg")
            negative_img = open_rgb(negative_img_path)

            if self.allow_transforms:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            else: 
                 # Apply only necessary transforms for evaluation
                 eval_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                 anchor_img = eval_transforms(anchor_img)
                 positive_img = eval_transforms(positive_img)
                 negative_img = eval_transforms(negative_img)


            return anchor_img, positive_img, negative_img
    
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
   
    def transform(self, img):
        """transform function"""
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
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