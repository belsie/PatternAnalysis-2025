"""
This file should show an example usage of your trained model. Print out any results 
and / or provide visualisations where applicable
"""
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import torch
import os
from train import train_funct
import global_vars as gv
from modules import Network
from dataset import MelanomaDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns

def test(test_loader, model, distance_threshold=0.5): 
    model.eval() 
    all_labels = []
    all_predictions = []
    all_distances = [] 

    with torch.no_grad():
        for step, (img1, img2, label) in enumerate(test_loader, 1): 
            img2 = img2.to(gv.DEVICE, non_blocking=True)
            label = label.to(gv.DEVICE, non_blocking=True)

            # Get embeddings
            embedding1 = model(img1)
            embedding2 = model(img2)

            # Calculate Euclidean distance between embeddings
            distance = torch.pairwise_distance(embedding1, embedding2)

            # Predict based on distance threshold
            prediction = (distance < distance_threshold).long() # 1 if similar, 0 if not

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(prediction.cpu().numpy())
            all_distances.extend(distance.cpu().numpy())


    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()

    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")


    return all_labels, all_predictions, all_distances

def plot_cm(test_labels, test_predictions):
    cm = confusion_matrix(test_labels, test_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Dissimilar', 'Predicted Similar'], yticklabels=['Actual Dissimilar', 'Actual Similar'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_loss(train_losses, val_losses, e = gv.EPOCHS):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, e + 1), train_losses, label='Training Loss')
    plt.plot(range(1, e + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--root", type=str, default=gv.ROOT)
    parser.add_argument("-bs","--batch-size", type=int, default=32)
    parser.add_argument("-nw","--num-workers", type=int, default=0)
    parser.add_argument('-lr', '--learning-rate', default= 1e-3)
    parser.add_argument("-e","--epochs", type=int, default=5)
    parser.add_argument("-m","--margin", type=int, default=0.4)
    args = parser.parse_args()

    if args.epochs is not None:
        gv.EPOCHS = int(args.epochs)

    data = MelanomaDataset(args.root)

    train_data, val_data, test_data = data.split_data(data.metadata, args.seed)

    train_dataset = MelanomaDataset(args.root, metadata=train_data)
    sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=int(len(train_dataset) * train_dataset.oversample_ratio), 
        replacement=True
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size,
        sampler=sampler, 
        num_workers=args.num_workers,
        pin_memory=True
    )


    val_loader = DataLoader(
        dataset = MelanomaDataset(args.root, metadata=val_data), 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_dataset = MelanomaDataset(args.root, metadata=test_data, is_test=True) 
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    train_losses, val_losses = train_funct(train_loader, val_loader, margin=args.margin)
    plot_loss(train_losses, val_losses, e = args.epochs)
    model = Network().to(gv.DEVICE)
    save_path = os.path.join(args.ROOT, "models", 'model_final.pth') 
    model.load_state_dict(torch.load(save_path))
    test_labels, test_predictions, test_distances = test(test_loader, model)
    plot_cm(test_labels, test_predictions)
    

if __name__ == "__main__":
    main()