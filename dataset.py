import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class WebsegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, rgb=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.rgb = rgb

        self.mappings = {
            (255,255,255): 0,   # white
            (255,0,0): 1,       # red
            (0,255,0): 2,       # green
            (0,0,255): 3,       # blue
            (255,0,255): 4,     # pink
        }

        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])

        if self.rgb:
            mask_path = os.path.join(self.mask_dir, self.images[item])
            mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)
        else:
            mask_path = os.path.join(self.mask_dir, self.images[item])
            mask = np.load(mask_path).astype(np.uint8)

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]

        return image, mask.permute(2, 0, 1)


def test():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
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

    data = WebsegDataset("data/train_frames", "data/train_masks_proc", rgb=False, transform=train_transform)
    print(len(data))
    img, mask = data[0]
    print(img.shape)
    print(mask.shape)
    print()
    print(np.mean(mask.numpy()))
    return
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(mask_to_rgb(mask.permute(1, 2, 0).numpy(), pd.read_csv("data/segdict.csv")))
    axs[1].imshow(img.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    test()
