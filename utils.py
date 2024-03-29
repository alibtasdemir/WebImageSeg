import pandas as pd
from PIL import Image
from typing import Tuple

from dataset import WebsegDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import tqdm
import cv2
import albumentations as A


def rgb_to_mask(image: np.ndarray, datadict: pd.DataFrame) -> np.ndarray:
    """
    This function converts the rgb mask (H, W, 3) to a multichannel segmentation mask (H, W, N)
    where the N is the number of classes. Need a pandas Dataframe which includes the segmentation map (RGB). 
    The rows of the dataframe should include (Red, Green, Blue) values for each class.

    Ex. 
    Class,R,G,B
    class1,255,0,0
    class2,0,0,255   

    The image must be a numpy array and the returning mask will be a numpy array.

    Args:
        image (np.ndarray): RGB mask (HxWx3).
        datadict (pd.DataFrame): The mapping dictionary in format of pandas.DataFrame

    Returns:
        np.ndarray: Multi-channel mask where each channel represents a class. (HxWxC)
    """
    mask = np.zeros((image.shape[0], image.shape[1], datadict.shape[0])).astype('int')

    for index, row in datadict.iterrows():
        add_val = np.zeros((1, 1, datadict.shape[0]))
        add_val[0, 0, index] = 1
        mask[(image[:, :, 0] == row.r) & (image[:, :, 1] == row.g) & (image[:, :, 2] == row.b)] = add_val
    return mask


def onehot_to_rgb(singlech_mask: np.ndarray) -> np.ndarray:
    """
    This function converts single channel masks to RGB masks. Single channel mask (H, W) contains
    single class label for corresponding pixels. The function generates an RGB image by mapping
    class labels to corresponding RGB values.

    Args:
        singlech_mask (np.ndarray): Single channel mask. The mask contains only one channel. Each pixel
        has a class label.

    Returns:
        np.ndarray: RGB image mask. (HxWx3)
    """
    # Mapping from RGB values to class labels
    rgb_to_class = {
        (255, 255, 255): 0,  # white
        (255, 0, 0): 1,  # red
        (0, 255, 0): 2,  # green
        (0, 0, 255): 3,  # blue
        (255, 0, 255): 4,  # pink
    }
    # The inverse of mapping. class labels -> RGB
    class_to_rgb = {v: k for k, v in rgb_to_class.items()}
    # Create an empty image filled with 0s.
    image = np.zeros((singlech_mask.shape[0], singlech_mask.shape[1], 3)).astype('int')

    for cl in class_to_rgb.keys():
        red, green, blue = class_to_rgb[cl]
        add_val = np.asarray([red, green, blue])
        # Place corresponding RGB values to the pixel.
        image[singlech_mask[:, :] == cl] = add_val
    return image


def rgb_to_mask_folder(imagesFolder: str, datadict: pd.DataFrame, savedir="data/masks_processed/"):
    """This function converts the rgb mask (H, W, 3) to multi-channel segmentation mask (H, W, N) where
    N is the number of classes. Need a pandas Dataframe which includes the segmentation map (RGB). 
    The rows of the dataframe should include (Red, Green, Blue) values for each class.

    Ex. 
    Class,R,G,B
    class1,255,0,0
    class2,0,0,255  

    This function processes a whole folder with rgb masks and saves the processed mask to another
    folder.

    Args:
        imagesFolder (str): The root folder for the RGB masks.
        datadict (pd.DataFrame): The mapping dictionary in format of pandas.DataFrame
        savedir (str, optional): Save path of the processed RGB masks. Defaults to "data/masks_processed/".
    """
    imgs = os.listdir(imagesFolder)
    tk = tqdm(imgs, total=len(imgs))
    for imgname in tk:
        imgpath = os.path.join(imagesFolder, imgname)
        img = Image.open(imgpath)
        img = np.asarray(img)
        new_image = rgb_to_mask(img, datadict=datadict)
        output_filename = savedir + imgname
        with open(output_filename, 'wb') as f:
            np.save(f, new_image, allow_pickle=True)


def mask_to_rgb(mask: np.ndarray, datadict: pd.DataFrame) -> np.ndarray:
    """
    This function converts a multichannel mask to rgb mask by using the given mapping. Reads channels
    and assigns mapped RGB values to generate RGB mask image (HxWx3).

    Args:
        mask (np.ndarray): Segmentation mask with multi-channels. (HxWxC) where C is the number of classes
        datadict (pd.DataFrame): The mapping dictionary in format of pandas.DataFrame

    Returns:
        np.ndarray: RGB mask (HxWx3)
    """
    image = np.zeros((mask.shape[0], mask.shape[1], 3)).astype('int')
    for index, row in datadict.iterrows():
        addval = np.asarray([row.r, row.g, row.b])
        image[mask[:, :, index] == 1] = addval
    return image


def mask_to_rgb_folder(imagesFolder: str, datadict: pd.DataFrame, save_dir="data/masks_processed/"):
    """This function converts multichannel masks to rgb masks by using the given mapping. Reads channels
    and assigns mapped RGB values to generate RGB mask image (HxWx3). Processes whole folder with multichannel
    masks and saves the output (RGB Masks) to a folder.

    Args:
        imagesFolder (str): The root folder for the multichannel masks.
        datadict (pd.DataFrame): The mapping dictionary in format of pandas.DataFrame
        save_dir (str, optional): Save path of the processed RGB masks. Defaults to "data/masks_processed/".
    """
    masks = os.listdir(imagesFolder)
    tk = tqdm(masks, total=len(masks))
    for mask in tk:
        mask_path = os.path.join(imagesFolder, mask)
        read_mask = np.load(mask_path)
        image = mask_to_rgb(read_mask, datadict=datadict)
        output_filename = save_dir + mask + '.png'
        cv2.imwrite(output_filename, image)


def get_loaders(
        train_dir: str,
        train_maskdir: str,
        val_dir: str,
        val_maskdir: str,
        batch_size: int,
        train_transform: A.Compose,
        val_transform: A.Compose,
        num_workers: int = 4,
        pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """This function generates dataloaders for train and validation sets.

    Args:
        train_dir (str): The path for the training image directory.
        train_maskdir (str): The path for the training segmentation mask directory.
        val_dir (str): The path for the validation image directory.
        val_maskdir (str): The path for the validation segmentation mask directory.
        batch_size (int): The batch size for the data loaders.
        train_transform (A.Compose): The transformation which will be applied to the training set. (albumentations)
        val_transform (A.Compose): The transformation which will be applied to the validation set. (albumentations)
        num_workers (int, optional): Number of workers. Defaults to 4.
        pin_memory (bool, optional): Pin memory flag value. Defaults to True.

    Returns:
        tuple[DataLoader, DataLoader]: Returns training data loader and validation data loader.
    """
    train_ds = WebsegDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = WebsegDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def pixel_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> np.float64:
    """This function calculates the pixel accuracy. Pixel accuracy denotes the correctly predicted
    pixels among the all pixels.

    Args:
        y_true (torch.Tensor): Ground-truth image tensor
        y_pred (torch.Tensor): Predicted image tensor

    Returns:
        np.float64: The accuracy value
    """
    y_pred = torch.argmax(y_pred, dim=1).float()
    y_pred = y_pred.unsqueeze(1)
    y_true = torch.argmax(y_true, dim=1).float().unsqueeze(1)

    accs = []
    for yt, yp in zip(torch.split(y_true, 1), torch.split(y_pred, 1)):
        yt, yp = yt.squeeze().cpu().numpy(), yp.squeeze().cpu().numpy()
        acc = (yt == yp).mean()
        accs.append(acc)
    return np.array(accs).mean()


def save_checkpoint(state: dict, filename: str = "checkpoint.pth.tar"):
    """Saves the torch model to the given path.

    Args:
        state (dict): A dictionary which includes the last state of the model and optimizer.
        filename (str, optional): The save path for the model. Defaults to "checkpoint.pth.tar".
    """
    print("=> Saving checkpoint.")
    torch.save(state, filename)


def load_checkpoint(checkpoint: dict, model: torch.nn.Module):
    """Loads the state from the checkpoint.

    Args:
        checkpoint (dict): A dictionary which includes the last state of the model and optimizer
        model (torch.nn.Module): The deep learning model
    """
    print("=> loading checkpoint.")
    model.load_state_dict(checkpoint["state_dict"])


def tb_save_image(
        writer: SummaryWriter, loader: DataLoader, model: torch.nn.Module, epoch: int, device: str = "cuda",
        max_images: int = 10
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx == 10:
            return
        x = x.to(device=device)
        x = x[0, None, ...]
        y = y[0, None, ...]

        with torch.inference_mode():
            preds = model(x)
            preds = torch.argmax(preds, dim=1).float()
            preds = preds.unsqueeze(1)
            y = torch.argmax(y, dim=1).float().unsqueeze(1)
            all_preds = []
            all_masks = []
            for mask, pred in zip(torch.split(y, 1), torch.split(preds, 1)):
                mask = mask.squeeze()
                pred = pred.squeeze()

                mask = torch.Tensor(onehot_to_rgb(mask.cpu().numpy())).permute(2, 0, 1)
                pred = torch.Tensor(onehot_to_rgb(pred.cpu().numpy())).permute(2, 0, 1)

                all_masks.append(mask)
                all_preds.append(pred)

            preds = torch.stack(all_preds) / 255.0
            y = torch.stack(all_masks) / 255.0

            x, preds, y = x.squeeze().cpu(), preds.squeeze().cpu(), y.squeeze().cpu()
            sum = torchvision.utils.make_grid([x, y, preds])
            # sum = np.concatenate((torch.squeeze(x).cpu(), torch.squeeze(preds).cpu(), torch.squeeze(y).cpu()), axis=1) # tf.squeeze deletes the batch dimension
            writer.add_image(f"Outputs_B{idx}", sum, global_step=epoch, dataformats="CHW")


def save_predictions_as_imgs(
        loader: DataLoader, model: torch.nn.Module, folder: str = "saved_images/", device: str = "cuda"
):
    """Saves models' prediction as images to the given folder. 

    Args:
        loader (DataLoader): Dataloader
        model (torch.nn.Module): The deep learning model
        folder (str, optional): The path for the save directory. Defaults to "saved_images/".
        device (str, optional): The device. It can be "cpu" if "cuda" is not enabled. Defaults to "cuda".
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.inference_mode():
            preds = model(x)
            preds = torch.argmax(preds, dim=1).float()
            preds = preds.unsqueeze(1)
            y = torch.argmax(y, dim=1).float().unsqueeze(1)
            # print(preds.shape)
            # preds = (preds > 0.5).float()
            all_preds = []
            all_masks = []
            for mask, pred in zip(torch.split(y, 1), torch.split(preds, 1)):
                mask = mask.squeeze()
                pred = pred.squeeze()

                mask = torch.Tensor(onehot_to_rgb(mask.cpu().numpy())).permute(2, 0, 1)
                pred = torch.Tensor(onehot_to_rgb(pred.cpu().numpy())).permute(2, 0, 1)

                all_masks.append(mask)
                all_preds.append(pred)

            preds = torch.stack(all_preds)
            y = torch.stack(all_masks)

        predpath = os.path.join(folder, f"pred_{idx}.png")
        maskpath = os.path.join(folder, f"{idx}.png")
        torchvision.utils.save_image(
            preds, predpath
        )
        torchvision.utils.save_image(y, maskpath)

    model.train()


def plot_training(
        training_losses: list,
        validation_losses: list,
        training_accuracy: list,
        validation_accuracy: list,
        learning_rate: list,
        gaussian: bool = True,
        sigma: bool = 2,
        figsize: Tuple[int, int] = (12, 6)
):
    """Creates plot for the training with training and validation losses and the learning rate.

    Args:
        training_losses (list): Training losses per epoch
        validation_losses (list): Validation losses per epoch
        training_accuracy (list): Training accuracy per epoch
        validation_accuracy (list): Validation accuracy per epoch
        learning_rate (list): Learning rates per epoch
        gaussian (bool, optional): Defaults to True.
        sigma (bool, optional): Defaults to 2.
        figsize (tuple[int, int], optional): Defaults to (12, 6).

    Returns:
        Returns the figure.
    """

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0, 0])
    subfig2 = fig.add_subplot(grid[0, 1])
    subfig3 = fig.add_subplot(grid[0, 2])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)
        training_accuracy_gauss = gaussian_filter(training_accuracy, sigma=sigma)
        validation_accuracy_gauss = gaussian_filter(validation_accuracy, sigma=sigma)

        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.25
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    # Subfig 1
    subfig1.plot(
        x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
        alpha=alpha
    )
    subfig1.plot(
        x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
        alpha=alpha
    )
    if gaussian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation loss')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')

    subfig1.legend(loc='upper right')

    # Subfig 2
    subfig2.plot(
        x_range, training_accuracy, linestyle_original, color=color_original_train, label='Training',
        alpha=alpha
    )
    subfig2.plot(
        x_range, validation_accuracy, linestyle_original, color=color_original_valid, label='Validation',
        alpha=alpha
    )
    if gaussian:
        subfig2.plot(x_range, training_accuracy_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig2.plot(x_range, validation_accuracy_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig2.title.set_text('Training & validation accuracy')
    subfig2.set_xlabel('Epoch')
    subfig2.set_ylabel('Accuracy')
    subfig2.set_ylim((0, 1))

    subfig2.legend(loc='lower right')

    # Subfig 3
    subfig3.plot(x_range, learning_rate, color='black')
    subfig3.title.set_text('Learning rate')
    subfig3.set_xlabel('Epoch')
    subfig3.set_ylabel('LR')

    return fig


def diceScore_multilabel(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int = 5, smooth: float = 0.0001):
    y_pred = torch.argmax(y_pred, dim=1).float()
    y_pred = y_pred.unsqueeze(1)
    y_true = torch.argmax(y_true, dim=1).float().unsqueeze(1)

    dices = []
    for yt, yp in zip(torch.split(y_true, 1), torch.split(y_pred, 1)):
        dice = 0
        yt, yp = yt.squeeze().cpu().numpy(), yp.squeeze().cpu().numpy()
        for i in range(n_classes):
            mask_yt = (yt == i)
            mask_yp = (yp == i)
            intersection = np.sum(mask_yt * mask_yp)
            dice += (2. * intersection * smooth) / (np.sum(mask_yt) + np.sum(mask_yp) + smooth)
            # print(f"Index: {i}\n\tYT pixels: {np.sum(mask_yt)}\n\tYP pixels: {np.sum(mask_yp)}\n\tIntersect: {intersection}\n\tDice: {dice}")
        dices.append(dice / n_classes)

    # print(dices)
    return np.array(dices).mean()


def dice_loss(logits, true, eps=1e-7, device="cuda"):
    from torch.nn import functional as F

    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    true = true.cpu()
    logits = logits.cpu()
    if true.shape[1] != 1:
        true = torch.argmax(true, dim=1)
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss).cuda()


def tverskyLoss(y_pred, y_true, alpha=0.3, beta=0.7, eps=1e-6):
    """
    y_pred: predicted tensor
    y_true: target tensor
    alpha: Tversky index coefficient
    beta: Tversky index coefficient
    eps: epsilon value for numerical stability
    """
    # flatten the tensors to 1D
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # true positive, false positive, false negative
    tp = torch.sum(y_pred * y_true)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp

    # Tversky index
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)

    # Tversky loss
    loss = 1 - tversky

    return loss


# TEST FUNCTIONS BELOW

def test():
    import matplotlib.pyplot as plt

    segdict = pd.read_csv("data/segdict.csv")

    train_mask_proc = "data/train_masks_proc/"
    val_mask_proc = "data/val_masks_proc/"

    train_masks = "data/train_masks/"
    val_masks = "data/val_masks/"

    # Convert rgb masks to multichannel
    rgb_to_mask_folder(train_masks, segdict, savedir=train_mask_proc)
    rgb_to_mask_folder(val_masks, segdict, savedir=val_mask_proc)

    # Check masks

    masks = os.listdir(train_mask_proc)
    print(os.path.join(train_mask_proc, masks[1]))
    readmask = np.load(os.path.join(train_mask_proc, masks[1]))
    plt.imshow(mask_to_rgb(readmask, segdict))
    plt.show()


def pixelTest():
    import glob
    def split_images(images, n=8):
        images = images[2:-2, :, :]
        image_list = []
        for i in range(n):
            starty = 2 + (i) * 258
            endy = starty + 256
            image_list.append(images[:, starty:endy, :])

        return image_list

    def all_split(image_paths):
        all_images = []
        for impath in image_paths:
            img = np.array(Image.open(impath).convert("RGB"))
            n = int((img.shape[1] - 2) / 258)
            all_images.extend(split_images(img, n=n))
        return all_images

    def accuracy_for_p(y, yhat):
        comp = yhat == y
        acc = (comp[:, :, 0] & comp[:, :, 1] & comp[:, :, 2]).mean()
        return acc

    pred_images = sorted(glob.glob(os.path.join("saved_images", "test1", "epoch_40", "pred_*.png")))
    gt_images = glob.glob(os.path.join("saved_images", "test1", "epoch_40", "*.png"))
    gt_images = sorted(list(set(gt_images) - set(pred_images)))

    preds = all_split(pred_images)
    gts = all_split(gt_images)

    accs = []
    for gt, pred in zip(gts, preds):
        accs.append(accuracy_for_p(gt, pred))
    accs = np.array(accs)
    print(f"Real accuracy is: {accs[:2].mean()}")

    import pandas as pd
    data = pd.read_csv("data/segdict.csv")

    yp = torch.stack([
        torch.Tensor(rgb_to_mask(preds[0], data).transpose([2, 0, 1])),
        torch.Tensor(rgb_to_mask(preds[1], data).transpose([2, 0, 1]))
    ])
    yt = torch.stack([
        torch.Tensor(rgb_to_mask(gts[0], data).transpose([2, 0, 1])),
        torch.Tensor(rgb_to_mask(gts[1], data).transpose([2, 0, 1])),
    ])
    print("Input to the function ---")
    print(f"shape yp: {yp.shape}\nshape yt: {yt.shape}")
    print("--- Inside function")
    print(f"The function accuracy is: {pixel_accuracy(yt, yp)}")


def plot_test():
    tloss = [1.3, 1.2, 0.8, 0.6, 0.5]
    vloss = [1.8, 1.4, 0.9, 0.8, 0.77]
    tacc = [0.3, 0.35, 0.56, 0.76, 0.85]
    vacc = [0.13, 0.21, 0.48, 0.66, 0.75]
    lr = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    fig = plot_training(tloss, vloss, tacc, vacc, lr, sigma=1)
    plt.show()


def test_diceLoss():
    from model import UNET
    from dataset import WebsegDataset
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from losses import DiceLoss

    train_transform = A.Compose(
        [
            A.Resize(height=256, width=256),
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

    data = WebsegDataset("data/1024/train_frames", "data/1024/train_masks_proc", rgb=False, transform=train_transform)
    model = UNET(in_channels=3, out_channels=5).to("cuda")

    print(len(data))
    img1, mask1 = data[0]
    img2, mask2 = data[1]

    img = torch.stack((img1, img2))
    mask = torch.stack((mask1, mask2))
    mask = mask.float().to("cuda")

    del img1, mask1, img2, mask2
    # print(img.shape)
    # print(mask.shape)

    pred = model(img.to("cuda"))
    loss_fn = lambda y_pred, y_true: tverskyLoss(y_pred, y_true, 0.5, 0.5)
    print(pred)
    loss = loss_fn(pred, mask)
    print(loss.item())
    print(loss)
    scaler = torch.cuda.amp.GradScaler()
    scaler.scale(loss).backward()


if __name__ == "__main__":
    # test()
    # pixelTest()
    # plot_test()
    test_diceLoss()

    """
    yt = torch.randint(0, 4, (5, 5, 5, 5), dtype=torch.long)
    yp = torch.randint(0, 4, (5, 5, 5, 5), dtype=torch.float, requires_grad=True)
    #yt = torch.argmax(yt, dim=1)
    from losses import DiceLoss
    lf = DiceLoss()
    lf = dice_loss
    loss = lf(yt, yp)
    scaler = torch.cuda.amp.GradScaler()
    print(loss.item())
    print(loss)
    scaler.scale(loss).backward()
    """
