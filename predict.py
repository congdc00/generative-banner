from PIL import Image
import numpy as np
import cv2

import pandas as pd
from core.stable_diffusion import SD_Model
from core.pix_art import PixArt_Model
from utils.enhance_image import Enhancer
IMG_SIZE = (1024, 536)

PRIVATE_PATH  = "/data/private/"
INPUT_PATH = f"{PRIVATE_PATH}info.csv"

SUBMISSION1_PATH = "/results/submission1/images"
SUBMISSION2_PATH = "/results/submission2/images"

class Model():

    def __init__(self, model_id):
        if model_id == "model_1":
            self.model = SD_Model(DEVICE, IMG_SIZE, "sd15")
        else:
            self.model = PixArt_Model(DEVICE, IMG_SIZE)

    def generate(self, prompt):
        positive_promt = prompt
        negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
        image = model.gen(positive_promt, negative_prompt=negative_prompt)
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
    for info, name_img in zip(list_info, bannerImage):
        prompt = preprocess(info)
        image = model.generate(prompt)
        image = model.enhanced(image)
        image.save(f"{SUBMISSION1_PATH}/{name_img}")

