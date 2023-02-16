import pandas as pd
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import pickle


def process_masks(imagesFolder, datadict, savedir="data/masks_processed/"):
    imgs = os.listdir(imagesFolder)
    tk = tqdm(imgs, total=len(imgs))
    for imgname in tk:
        imgpath = os.path.join(imagesFolder, imgname)
        img = Image.open(imgpath)
        img = np.asarray(img)
        new_image = np.zeros((img.shape[0],img.shape[1], datadict.shape[0])).astype('int')

        for index, row in datadict.iterrows():
            add_val = np.zeros((1, 1, 5))
            add_val[0, 0, index] = 1
            new_image[(img[:,:,0]==row.r) & (img[:,:,1]==row.g) & (img[:,:,2]==row.b)] = add_val
        
        # print(f"Avg. 1 channel: {new_image[:,:,0].mean()}")
        # print(f"Avg. 2 channel: {new_image[:,:,1].mean()}")
        # print(f"Avg. 3 channel: {new_image[:,:,2].mean()}")
        # print(f"Avg. 4 channel: {new_image[:,:,3].mean()}")
        # print(f"Avg. 5 channel: {new_image[:,:,4].mean()}")
        # print(f"Shape of the new mask: {new_image.shape}")
        
        output_filename = savedir + imgname+'.png'
        with open(output_filename, 'wb') as f:
            np.save(f, new_image, allow_pickle=True)
        # cv2.imwrite(output_filename,new_image)


train_masks = "data/train_masks/"
val_masks = "data/val_masks/"
segdict = pd.read_csv("data/segdict.csv")

process_masks(train_masks, segdict, savedir="data/train_masks_proc/")
process_masks(val_masks, segdict, savedir="data/val_masks_proc/")
