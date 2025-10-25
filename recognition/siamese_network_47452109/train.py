"""
Contains the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""
import global_vars as gv
from modules import Network, TripletLoss
import os
import torch
from torch.optim.adamw import AdamW
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import StepLR 

def train_funct(train_loader, val_loader, lr = 1e-3, accumulation_steps=4, margin=0.5): 
    model = Network().to(gv.DEVICE) 
    model.train()

    # Save at {path}/models/..
    save_dir = os.path.join(gv.ROOT, "models")
    os.makedirs(save_dir, exist_ok=True)

    loss_fn = TripletLoss(margin=margin) 

    optimizer = AdamW(model.parameters(), lr= lr)

    scaler = GradScaler(gv.DEVICE.type)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1) 

    print("Training start.")

    train_losses = [] 
    val_losses = [] 

    for epoch in range(0, gv.EPOCHS):

        running_loss = 0.0
        model.train() 
        for step, (anchor_img, positive_img, negative_img) in enumerate(train_loader,1): 
            anchor_img = anchor_img.to(gv.DEVICE, non_blocking = True)
            positive_img = positive_img.to(gv.DEVICE, non_blocking = True)
            negative_img = negative_img.to(gv.DEVICE, non_blocking = True)

            with autocast(device_type=gv.DEVICE.type, enabled=True):
                # compute embeddings for images
                anchor_embedding = model(anchor_img)
                positive_embedding = model(positive_img)
                negative_embedding = model(negative_img)

                # Calculate Triplet Loss
                loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)


            # Scale loss and backpropagate
            scaler.scale(loss).backward()

            # Perform optimizer step and zero gradients only after accumulating gradients
            if step % accumulation_steps == 0:
                scaler.step(optimizer) 
                scaler.update() 
                optimizer.zero_grad(True)

            running_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Train loss {running_loss/50:.4f}")
                running_loss = 0.0

        # Perform optimizer step and zero gradients for the last mini-batches if they don't
        # form a full accumulation_steps batch
        if (step % accumulation_steps != 0):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(True)

        # Record training loss for the epoch
        epoch_train_loss = running_loss / (len(train_loader) % 50 if len(train_loader) % 50 != 0 else 50) # Calculate average for the last part
        train_losses.append(epoch_train_loss)

        scheduler.step() 

        # Run validation after each epoch
        avg_val_loss = validate(model, val_loader, loss_fn, epoch) 
        val_losses.append(avg_val_loss) 

        # Save model after each epoch
        save_path_epoch = os.path.join(save_dir, f'model2_epoch_{epoch}.pth')
        torch.save(model.state_dict(), save_path_epoch)
        print(f"Model saved after epoch {epoch} at {save_path_epoch}")


    save_path = os.path.join(save_dir, 'model_final.pth') 
    torch.save(model.state_dict(), save_path)
    print("Training complete. Final model saved.")

    return train_losses, val_losses 


def validate(model, val_loader, loss_fn, epoch):
    model.eval() 
    running_loss = 0.0

    with torch.no_grad():
        with autocast(device_type=gv.DEVICE.type, enabled=True):
            for step, (anchor_img, positive_img, negative_img) in enumerate(val_loader, 1): 
                anchor_img = anchor_img.to(gv.DEVICE, non_blocking=True)
                positive_img = positive_img.to(gv.DEVICE, non_blocking=True)
                negative_img = negative_img.to(gv.DEVICE, non_blocking=True)

                # compute embeddings for images
                anchor_embedding = model(anchor_img)
                positive_embedding = model(positive_img)
                negative_embedding = model(negative_img)

                # Calculate Triplet Loss
                loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
                running_loss += loss.item()

    avg_loss = running_loss / len(val_loader)
    print(f"Epoch {epoch} | Validation loss {avg_loss:.4f}")

    return avg_loss 

    
        
