"""
Contains the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""
from modules import SiameseNetwork, TripletLoss
from dataset import MelanomaDataset
import global_vars as gv
import os
import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
import argparse

def train(train_loader, lr = 1e-3):    
    model = SiameseNetwork().to(gv.DEVICE)
    model.train()

    # Save at {path}/models/..
    save_dir = os.path.join(gv.ROOT, "models")
    os.makedirs(save_dir, exist_ok=True)

    #loss_fn = TripletLoss(margin=0.5)
    crit = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= lr)
    #TODO: scheduler

    print("Training start.")
    for epoch in range(0, gv.EPOCHS):
        running_loss = 0.0
        for step, (img1,img2,y) in enumerate(train_loader,1):
            img1 = img1.to(gv.DEVICE, non_blocking = True)
            img2 = img2.to(gv.DEVICE, non_blocking = True)
            y = y.float().to(gv.DEVICE, non_blocking = True)

            e1, e2 = model(img1, img2)
            
            cos = nn.functional.cosine_similarity(e1, e2).unsqueeze(1)
            # scale by 3
            logit = 3.0 * cos.squeeze(1)
            # backward and optimize
            loss = crit(logit, y)
            optimizer.zero_grad(True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | loss {running_loss/50:.4f}")
                running_loss = 0.0

    save_path = os.path.join(save_dir, 'model.pth')
    torch.save(model.state_dict(), save_path)
    print("Training complete. Models saved.")

def validate():
    return

def test():
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--root", type=str, default=gv.ROOT)
    parser.add_argument("-bs","--batch-size", type=int, default=8)
    parser.add_argument("-nw","--num-workers", type=int, default=0)
    parser.add_argument('-lr', '--learning-rate', default= 1e-3)
    parser.add_argument("-e","--epochs", type=int, default=5)
    args = parser.parse_args()

    if args.epochs is not None:
        gv.EPOCHS = int(args.epochs)

    data = MelanomaDataset(args.root)
    data.set_subset(100,seed= 123)
    loader = DataLoader(
        dataset = data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train(loader)


if __name__ == "__main__":
    main()

    
        
