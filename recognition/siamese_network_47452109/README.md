# Using a Siamese Network for Classification of Melanoma
Siamese networks consist of two or more networks working synchronously using shared weights. These networks use metric learning, meaning they learn a distance function that measures how similar or dissimilar two input samples are, and is ideal for identifying subtly differences between classes, such as distinguishing beween malignant and benign lesions in medical imaging tasks. 

## Dataset
The *SIIM-ISIC Melanoma Classification* dataset provides over 33,000 dermoscopic images of lesions with a known melanoma diagnosis. Each lesion is identified as either:
- **benign / non-cancerous** (labelled by 0)
- **malignant / melanoma** (labelled by 1)

For more information, view the Kaggle page [here](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview). 

## Problem
The challenge is to train a Siamese Network on the SIIM-ISIC Melanoma Classification dataset such that by the end of training and validation, it can identify a lesion image with around 80% accuracy. However, the dataset its highly imbalanced, with over 98% of the samples being benign. This is a problem as limited malignant examples restrict the model’s ability to learn meaningful patterns of the minority class, which results in bias toward the majority (benign) class.

### Preprocessing
To counteract the data imbalance, techniques such as data augmentation, oversampling the minority class (and undersampling the majority) were used. 

The following augemntation methods were utilised:
1. Resizing of the data to 224x224 pixel images.
3. Normalisation of the pixel values using mean and standard deviation.
4. Randomly flipping the image horizontally or vertically (p = 0.5 for both).
5. Rotating 15 degrees randomly
6. Resizing (for consistancy)

These methods were chosen so that the neural network could identify skin lesions more generally. Colour jitter was omitted as the colour of a lesion is important when identifying it. 

Furthermore, the malignant class was oversampled by using calculated weights based on how many of each class are in the dataset. This assists the network to learn each type of class equally.

As there is no publicly available test data labels, test data was manufactured from the provided training data. The training data followed a 75:15:10 split (training | validation | test) to maximise learning while preserving enough data for validation and testing such that the results aren't too noisy. This model utilises triplet loss to calculate loss:

L=max{0, d(a,p)−d(a,n)+m}

where a is the anchor image, p is a positive of the same class, n is  a negative of a different class, d() is a distance (L2 or 1−cosine), and 
m is a margin. This forces embeddings of the same diagnosis to be closer than different diagnoses by at lease m. 

## Using this algorithm
To successfully use this algorithm, first ensure that the correct file structure is set up (see below).
```
root
  ├─ README.md
  ├─ global_vars.py
  ├─ modules.py
  ├─ dataset.py
  ├─ train.py
  ├─ predict.py
  └─ data
    ├─ ISIC_2020_Training_GroundTruth_v2.csv
    ├─ train
    │ ├─ ISIC_0015719.jpg
    │ ├─ ...
    │ └─ ISIC_9999806.jpg
    └─ models (created at runtime by training)
        ├─ siamese1.pt
        └─ ...
```
The algorithm can be initiated using the terminal using the arguments below. 

`--root` `-r`: the root directory. No default, must be included.  \
`--batch-size` `-bs`: batch size. Default = 8. \
`--num-workers` `-nw`: DataLoader workers. Default = 0. \
`--learning-rate` `-lr`: learning rate. Default = 1e-3. \
`--epochs` `-e`: Amount of epochs in training. Default = 5. \
`--margin` `-m`: Margin used in triplet loss. Default = 0.4. \
`--seed` `-s`: seed for reproducablilty. 

See below for an example input.
```
python -m \
  --root \path\file \
  --batch-size 32 \
  --num-workers 4 \
  --learning-rate 1e-3 \
  --epochs 10 \
  --seed 123
```
Once the training loop is complete, the model will be saved to /root/models/ for easy access. 
### Dependencies
`python 3.11.9` \
`torch 2.4.1+cu118` \
`torchvision 0.19.1+cu118` \
`pandas 2.2.1` \
`pillow 10.2.0` \
`scikit-learn 1.5.0` 

Note: Code only tested using above dependencies.

### Outputs
After training, a graphic of training and validation loss over each epoch is produced.

After testing, a confusion matrix is produced as well as summary statistics
<img width="695" height="590" alt="image" src="https://github.com/user-attachments/assets/493c9799-cf29-4c11-be42-34744d1f7360" />

(NOTE, results below after running one epoch only!!)
```
Test Results:
Accuracy: 0.5074
Precision: 0.5025
Recall: 0.9848
F1 Score: 0.6654
True Positives (TP): 1623
True Negatives (TN): 58
False Positives (FP): 1607
False Negatives (FN): 25
```


