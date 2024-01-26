import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
from early_stopping import EarlyStopping
import torch.optim as optim
from model import UNET, AttU_Net
from utils import get_loaders, pixel_accuracy, save_checkpoint, load_checkpoint, save_predictions_as_imgs, \
    plot_training, tb_save_image
from metric_monitor import MetricMonitor
import os
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# Hyperparameters etc.
MODELNAME = "Attention UNET"
# MODELNAME = "UNET"
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/256/train_frames/"
TRAIN_MASK_DIR = "data/256/train_masks_proc/"
VAL_IMG_DIR = "data/256/val_frames/"
VAL_MASK_DIR = "data/256/val_masks_proc/"

EXP_NAME = "log-attention-lr1e4"
EXP_ROOT = "experiments/" + str(IMAGE_HEIGHT)
EXP_SAVEDIR = os.path.join(EXP_ROOT, EXP_NAME)
TB_DIR = os.path.join(os.path.join(EXP_SAVEDIR, "logdir"))

# INIT DIRECTORIES
if not os.path.exists(EXP_SAVEDIR): os.makedirs(EXP_SAVEDIR)
# Create checkpoints dir
if not os.path.exists(os.path.join(EXP_SAVEDIR, "checkpoints")): os.makedirs(os.path.join(EXP_SAVEDIR, "checkpoints"))
# Create experiment results dir
if not os.path.exists(os.path.join(EXP_SAVEDIR, "saved_images")): os.makedirs(os.path.join(EXP_SAVEDIR, "saved_images"))
# Create tensorboard log dir
if not os.path.exists(os.path.join(EXP_SAVEDIR, "logdir")): os.makedirs(os.path.join(EXP_SAVEDIR, "logdir"))


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    train_losses = []
    accs = []
    metric_monitor = MetricMonitor()
    model.train()

    train_loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(train_loop):
        data = data.to(DEVICE)
        # targets = targets.float().unsqueeze(1).to(DEVICE)
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

    # model = UNET(in_channels=3, out_channels=5).to(DEVICE)
    model = AttU_Net(img_ch=3, output_ch=5).to(DEVICE)
    # loss_fn = nn.CrossEntropyLoss()
    from utils import dice_loss, tverskyLoss
    # from losses import TverskyLoss
    loss_fn = dice_loss
    # loss_fn = tverskyLoss
    # loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping = EarlyStopping(patience=50, verbose=True,
                                   path=os.path.join(EXP_SAVEDIR, "checkpoints", "checkpoint_best.pt"))

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

    writer = SummaryWriter(TB_DIR, flush_secs=5)

    hp_dict = {
        'Model Name': MODELNAME,
        'Optimizer': "Adam",
        'Epoch': NUM_EPOCHS,
        'Batch Size': BATCH_SIZE,
        'Image HxW': f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}",
        'Learning Rate': LEARNING_RATE
    }

    # configure(TB_DIR, flush_secs=5)

    training_losses = []
    validation_losses = []
    training_acc = []
    validation_acc = []

    for epoch in range(1, NUM_EPOCHS + 1):

        train_epoch_loss, train_epoch_acc = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch=epoch)
        training_losses.append(train_epoch_loss)
        training_acc.append(train_epoch_acc)

        # check acc
        val_epoch_loss, val_epoch_acc = validation_fn(val_loader, model, loss_fn, epoch=epoch)
        validation_losses.append(val_epoch_loss)
        validation_acc.append(val_epoch_acc)

        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Loss/validation', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_epoch_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_epoch_acc, epoch)

        # log_value('training_loss', train_epoch_loss, epoch)
        # log_value('validation_loss', val_epoch_loss, epoch)
        # log_value('training_accuracy', train_epoch_acc, epoch)
        # log_value('validation_accuracy', val_epoch_acc, epoch)

        early_stopping(val_epoch_loss, model)

        if early_stopping.early_stop:
            print("Stopping...")
            # model.load_state_dict(torch.load(early_stopping.path))
            break

        if (epoch % 10 == 0) or (epoch == 1):
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            chk_path = os.path.join(EXP_SAVEDIR, "checkpoints", f"checkpoint_att_{epoch}.pth.tar")
            save_checkpoint(checkpoint, filename=chk_path)

            img_save_folder = os.path.join(EXP_SAVEDIR, "saved_images", f"epoch_{epoch}")
            if not os.path.exists(img_save_folder):
                os.makedirs(img_save_folder)

            # print some examples to a folder
            save_predictions_as_imgs(val_loader, model, folder=img_save_folder, device=DEVICE)
            tb_save_image(writer, val_loader, model, epoch, device=DEVICE)

    writer.add_hparams(hp_dict, {
        'hparam/accuracy': validation_acc[-1],
        'hparam/train-loss': training_losses[-1],
        'hparam/val-loss': validation_losses[-1],
    })

    fig = plot_training(
        training_losses,
        validation_losses,
        training_acc,
        validation_acc,
        [LEARNING_RATE for _ in range(len(training_losses))],
        sigma=1,
        figsize=(15, 4)
    )
    writer.add_figure('training_val_plot', fig)

    plot_path = os.path.join(EXP_SAVEDIR, "saved_images", "training_plot.png")
    plt.savefig(plot_path)
    plt.show()


def findlr(model, loader):
    from torch_lr_finder import LRFinder
    from utils import dice_loss

    loss_fn = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-7)

    lr_finder = LRFinder(model, optimizer, loss_fn, device="cuda")
    lr_finder.range_test(loader, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph


if __name__ == "__main__":
    main()
