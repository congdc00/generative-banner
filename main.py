import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import cosine_similarity
from PIL import Image
import re
from translate import translate

import pandas as pd
device = "cuda"
IMG_SIZE = (1024, 533)
INPUT_PATH = "./data/task2/test/info.csv"
OUTPUT_PATH = "./data/task2/test/submission2/images/"
phoBert = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")
TRAIN_PATH = "./data/task2/train/info.csv"
 

def init_mode(model, vae):
    # vae_model = ConsistencyDecoderVAE.from_pretrained(vae, torch_dtype=pipe.torch_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        model, 
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    pipe.enable_attention_slicing()
    
    return pipe
def get_embedding(text):
    inputs = phoBert(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
def get_cosine(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def gen_img(prompt):
    image = pipe(prompt, width = IMG_SIZE[0], height = IMG_SIZE[1]+2).images[0]
    cropped_image = image.crop((0, 2, IMG_SIZE[0], IMG_SIZE[1]+2))
    return cropped_image

def get_img(prompt):
    info_train_path = TRAIN_DATA_PATH + "info.csv"
    print(f"\n prompt {prompt}")
    with open(info_train_path, 'r') as file:
        i = 0 
        max_point = 0

        for line in tqdm(file, desc="Processing", unit="file"):
            
            if '"' in line:
                caption = str(line.split('"')[1])
            else:
                caption = str(line.split(",")[1])
 
            bannerimage = str(line.split(",")[-1]).replace("/n", "")
            bannerimage = bannerimage.split('.')[0] + ".jpg"

            if i != 0:
                point = get_cosine(caption, prompt)
                if point > max_point:
                    image_path = TRAIN_DATA_PATH + "images/" + bannerimage
                    img = Image.open(image_path)
                    max_point = point
            i+= 1
    if max_point != 0:
        print(f"Load image {image_path}")
        return img
    else:
        return False



if __name__ == "__main__":
    
    # pipe = init_mode(model = "runwayml/stable-diffusion-v1-5", vae ="openai/consistency-decoder" )
    
    # test
    df_test = pd.read_csv(INPUT_PATH)
    captions_test = list(df_test['caption'])
    description_test = list(df_test['description'])
    moreInfo_test = list(df_test['moreInfo'])
    bannerImage_test = list(df_test['bannerImage'])

    # TRAIN 
    df_train = pd.read_csv(TRAIN_PATH)
    captions_train = list(df_train['caption'])
    description_train = list(df_train['description'])
    moreInfo_train = list(df_train['moreInfo'])
    bannerImage_train = list(df_train['bannerImage'])
    
    captions_train = phoBert(captions_train, return_tensors="pt", padding=True, truncation=True)["input_ids"].float()
    captions_test = phoBert(captions_test, return_tensors="pt", padding=True, truncation=True)["input_ids"].float()

    if captions_test.shape[1] > captions_train.shape[1]:
        padding_size = captions_test.shape[1] - captions_train.shape[1]
        captions_train = torch.nn.functional.pad(captions_train, (0, padding_size))

    elif captions_test.shape[1] < captions_train.shape[1]:
        padding_size = captions_train.shape[1] - captions_test.shape[1]
        captions_test = torch.nn.functional.pad(captions_test, (0, padding_size))
    ''''
    if captions_test.shape[0] > captions_train.shape[0]:
        padding_size = captions_test.shape[0] - captions_train.shape[0]
        captions_train = torch.nn.functional.pad(captions_train, (0, 0, 0, padding_size))

    elif captions_test.shape[0] < captions_train.shape[0]:
        padding_size = captions_train.shape[0] - captions_test.shape[0]
        captions_test = torch.nn.functional.pad(captions_test, (0, 0, 0, padding_size))
    '''
    captions_test = captions_test.unsqueeze(0).expand(len(captions_train), -1, -1)
    captions_train = captions_train.unsqueeze(0)
    captions_test = captions_test.unsqueeze(1)
    similarity_matrix = cosine_similarity(captions_test, captions_train, dim=1)
    
    print(similarity_matrix.shape)

    # output_path = f"{OUTPUT_PATH}{bannerimage}"
