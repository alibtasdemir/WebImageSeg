import pandas as pd
from PIL import Image
import os, glob
import numpy as np
from tqdm import tqdm
import pickle


def process_masks(imagesFolder, datadict, savedir="data/masks_processed/", clean_ext=None):
    imgs = glob.glob(os.path.join(imagesFolder, '*'))
    os.makedirs(savedir, exist_ok=True)
    tk = tqdm(imgs, total=len(imgs))
    for imgpath in tk:
        imgname = imgpath.split(os.sep)[-1]
        # imgpath = os.path.join(imagesFolder, imgname)
        try:
            img = Image.open(imgpath)
        except FileNotFoundError as e:
            print("File not found: {}".format(imgpath))
            continue
        img = np.asarray(img)
        new_image = np.zeros((img.shape[0], img.shape[1], datadict.shape[0])).astype('int')

        for index, row in datadict.iterrows():
            add_val = np.zeros((1, 1, 5))
            add_val[0, 0, index] = 1
            new_image[(img[:, :, 0] == row.r) & (img[:, :, 1] == row.g) & (img[:, :, 2] == row.b)] = add_val

        # print(f"Avg. 1 channel: {new_image[:,:,0].mean()}")
        # print(f"Avg. 2 channel: {new_image[:,:,1].mean()}")
        # print(f"Avg. 3 channel: {new_image[:,:,2].mean()}")
        # print(f"Avg. 4 channel: {new_image[:,:,3].mean()}")
        # print(f"Avg. 5 channel: {new_image[:,:,4].mean()}")
        # print(f"Shape of the new mask: {new_image.shape}")

        if clean_ext:
            imgname = imgname.replace(clean_ext, '')
        output_filename = savedir + imgname

        with open(output_filename, 'wb') as f:
            np.save(f, new_image, allow_pickle=True)
        # cv2.imwrite(output_filename,new_image)


train_masks = "data/256/train_masks/"
val_masks = "data/256/val_masks/"
segdict = pd.read_csv("data/segdict.csv")

process_masks(train_masks, segdict, savedir="data/256/train_masks_proc/", clean_ext="_L")
process_masks(val_masks, segdict, savedir="data/256/val_masks_proc/", clean_ext="_L")
