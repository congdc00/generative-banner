from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
from core.stable_diffusion import SD_Model
from core.pix_art import PixArt_Model
from utils.enhance_image import Enhancer
from preprocessing.preprocess import process

IMG_SIZE = (1024, 536)
DEVICE = "cuda"
PRIVATE_PATH  = "/home/congdc/hdd/Project/data/private/"
INPUT_PATH = f"{PRIVATE_PATH}info.csv"

SUBMISSION1_PATH = "/home/congdc/hdd/Project/results/submission1/images"
SUBMISSION2_PATH = "/home/congdc/hdd/Project/results/submission2/images"

class Model():

    def __init__(self, model_id):
        if model_id == "model_1":
            self.model = SD_Model(DEVICE, IMG_SIZE, "sd15_ver2")
        else:
            self.model = SD_Model(DEVICE, IMG_SIZE, "sd15_ver1")

    def generate(self, prompt):
        positive_promt = prompt
        negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
        image = self.model.gen(positive_promt, negative_prompt=negative_prompt)
        image = image.crop((0, 1, 1024, 534))
        
        return image

    def enhanced(self, image):
        image = Enhancer.enhance(image)
        return image

 
if __name__ == "__main__":
    
    # data
    df = pd.read_csv(INPUT_PATH)    
    captions = list(df['caption'])
    description = list(df['description'])
    moreInfo = list(df['moreInfo'])
    list_info = zip(captions, description, moreInfo)
    bannerImage = list(df['bannerImage'])
    
    # model 1
    model = Model("model_1")
    os.makedirs(SUBMISSION1_PATH, exist_ok = True)
    for info, name_img in zip(list_info, bannerImage):
        prompt = process(info)
        image = model.generate(prompt)
        image = model.enhanced(image)
        image.save(f"{SUBMISSION1_PATH}/{name_img}")

    # model 2
    model = Model("model_2")
    os.makedirs(SUBMISSION2_PATH, exist_ok = True)
    for info, name_img in zip(list_info, bannerImage):
        prompt = process(info)
        image = model.generate(prompt)
        image = model.enhanced(image)
        image.save(f"{SUBMISSION1_PATH}/{name_img}")
