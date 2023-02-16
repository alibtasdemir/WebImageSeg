import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import get_loaders, pixel_accuracy, save_checkpoint, load_checkpoint, save_predictions_as_imgs, plot_training
from metric_monitor import MetricMonitor
import os
import matplotlib.pyplot as plt

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_frames/"
TRAIN_MASK_DIR = "data/train_masks_proc/"
VAL_IMG_DIR = "data/val_frames/"
VAL_MASK_DIR = "data/val_masks_proc/"


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    train_losses = []
    accs = []
    metric_monitor = MetricMonitor()
    model.train()

    train_loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(train_loop):
        data = data.to(DEVICE)
        #targets = targets.float().unsqueeze(1).to(DEVICE)
        targets = targets.float().to(DEVICE)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        loss_val = loss.item()
        train_losses.append(loss_val)
        
        accuracy_val = pixel_accuracy(targets, predictions)
        accs.append(accuracy_val)
        
        # backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metric_monitor.update("Loss", loss_val)
        metric_monitor.update("Accuracy", accuracy_val)

        train_loop.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
        train_loop.set_postfix(loss=loss_val)
    
    return np.array(train_losses).mean(), np.array(accs).mean()


def validation_fn(loader, model, loss_fn, epoch):
    metric_monitor = MetricMonitor()
    model.eval()
    validation_losses = []
    accs = []
    
    validation_loop = tqdm(loader)

    with torch.inference_mode():
        for batch_idx, (data, targets) in enumerate(validation_loop):
            data = data.to(DEVICE)
            targets = targets.float().to(DEVICE)
            # forward
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
            loss_val = loss.item()
            accuracy_val = pixel_accuracy(targets, predictions)
            validation_losses.append(loss_val)
            accs.append(accuracy_val)
            metric_monitor.update("Loss", loss_val)
            metric_monitor.update("Accuracy", accuracy_val)

            validation_loop.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
    
    return np.array(validation_losses).mean(), np.array(accs).mean()



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=3, out_channels=5).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()

    training_losses = []
    validation_losses = []
    training_acc = []
    validation_acc = []

    for epoch in range(1, NUM_EPOCHS+1):

        train_epoch_loss, train_epoch_acc = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch=epoch)
        training_losses.append(train_epoch_loss)
        training_acc.append(train_epoch_acc)

        # check acc
        val_epoch_loss, val_epoch_acc = validation_fn(val_loader, model, loss_fn, epoch=epoch)
        validation_losses.append(val_epoch_loss)
        validation_acc.append(val_epoch_acc)

        if (epoch % 10 == 0) or (epoch == 1):
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"checkpoints/checkpoint_{epoch}.pth.tar")

            img_save_folder = os.path.join("saved_images", f"epoch_{epoch}")
            if not os.path.exists(img_save_folder):
                os.makedirs(img_save_folder)

            # print some examples to a folder
            save_predictions_as_imgs(val_loader, model, folder=img_save_folder, device=DEVICE)

    
    fig = plot_training(
        training_losses, 
        validation_losses, 
        training_acc, 
        validation_acc, 
        [LEARNING_RATE for i in range(len(training_losses))], 
        sigma=1, 
        figsize=(15, 4)
        )
    
    plt.savefig("training_plot.png")
    plt.show()        


if __name__ == "__main__":
    main()
