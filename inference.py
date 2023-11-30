from PIL import Image
import numpy as np
import cv2
from utils.translate import Translater
import pandas as pd
from core.stable_diffusion import SD_Model
DEVICE = "cuda"
IMG_SIZE = (1024, 536)
TEST_PATH  = "./data/test/"
INPUT_PATH = f"{TEST_PATH}info.csv"
OUTPUT_PATH = f"{TEST_PATH}images"
 
if __name__ == "__main__":
    
    # data
    df = pd.read_csv(INPUT_PATH)
    captions = list(df['caption'])
    description = list(df['description'])
    moreInfo = list(df['moreInfo'])
    bannerImage = list(df['bannerImage'])
    
    # model 
    model = SD_Model(DEVICE, IMG_SIZE)
    model.enhancer_model()

    for idx, (caption, name) in enumerate(zip(captions, bannerImage)):
        positive_promt = Translater.vie2eng(caption)
        positive_promt = f"a banner vietnamese, {positive_promt}"
        negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
        
        image = model.gen(positive_promt, negative_prompt=negative_prompt)
        image = image.crop((0, 1, 1024, 534))

        output_path = f"{OUTPUT_PATH}/{name}"
        image.save(output_path)

