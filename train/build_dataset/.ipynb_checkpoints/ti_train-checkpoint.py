from PIL import Image
import numpy as np
import cv2
from utils.translate import Translater
import pandas as pd
from core.stable_diffusion import SD_Model
import re
from collections import Counter
import nltk
from utils.translate import Translater
import underthesea

DEVICE = "cuda"
TRAIN_PATH  = "./data/train/target/"
INPUT_PATH = f"{TRAIN_PATH}info.csv"
OUTPUT_PATH = f"{TRAIN_PATH}images"

def keys_exist(list_keys, content):
    for keys in list_keys:
        is_exist = False
        for key in keys:
            if key in content:
                is_exist = True
                break
        if is_exist == True:
            continue
        else: 
            return False
    return is_exist

def extract_keyword(type_sentence, sentence):
    
    key_words = []

    list_map = [{"key":[["áo phủ", "bạt phủ"]], "value": ["car cover"]},
                {"key":[["thảm"], ["ô tô", "xe"]], "value": ["car floor mat"]},
                {"key":[[""], []]}
                ]
    
    for tur in list_map:
        if keys_exist(tur["key"], sentence):
            for value in tur["value"]:
                if value not in key_words:
                    key_words.append(value)
    return key_words

def find_keys(content):
    content = " ".join(content) 
    cleaned_text = re.sub(r'[^\w\s]', ' ', content)
    text_segment = underthesea.word_tokenize(cleaned_text, format="text")
    words = text_segment.split()

    word_counter = Counter(words)
    return word_counter



if __name__ == "__main__":
    
    # data
    df = pd.read_csv(INPUT_PATH)
    id_imgs = list(df["id"])
    captions = list(df['caption'])
    description = list(df['description'])
    moreInfo = list(df['moreInfo'])
    bannerImages = list(df['bannerImage'])

    keys_word_caption = find_keys(captions)
    print(f"keys_word_caption {keys_word_caption}")

    '''
    list_key_word = []
    i = 0
    for id_img, caption, desc, info, name_img in zip(id_imgs, captions, description, moreInfo, bannerImages):
        
        key_words = []

        caption = caption.lower()
        desc = desc.lower()
        info = info.lower()

        list_kw1 = extract_keyword("caption", caption)
        key_words += list_kw1
        
        list_key_word.append(key_words)

        print(f"CAPTION {caption}")
        print(f"KEYWORDS {key_words}")
        
        i+=1 
        if i == 2:
            break
    '''

    # save
    # df.drop(columns=["id", 'caption', 'description', "moreInfo"], inplace=True)
    # df = df.assign(key_words=id_imgs)
    # df.to_csv("./data_processing/ti_data.json", index=False, encoding='utf-8'),    

