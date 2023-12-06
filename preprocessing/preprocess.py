from utils.translate import Translater
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import re
from collections import Counter
import nltk
import underthesea
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import shutil
import transformers
import spacy

def clean_text(sentence):
    sentence = str(sentence).lower()
    sentence = sentence.replace(" - ", "-")
    # sentence = re.sub(r'[.\-\/]', ' ', sentence)
    return sentence

def translater(input):
    result = Translater.vie2eng(input)
    return result

def gen_prompt(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("but", " ")
    sentence = sentence.replace("also", " ")
    sentence = sentence.replace("-", " ")
    sentence = sentence.replace(".", ",")
    sentence = sentence.replace("en:", "")
    sentence = sentence.replace("+", "")
    sentence = sentence.replace(",,,", "")
    sentence = sentence.replace("%", "")
    sentence = sentence.replace("$", "")
    sentence = sentence.replace("!", ",")
    sentence = re.sub(r'\b\d\S*\b', 'x', sentence)
    sentence = re.sub(r'\b(\w+)\s+\1\b', r'\1', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    
    if sentence[-1] == ",":
        sentence = sentence[:-1]
    list_word = sentence.split(" ")
    counts = Counter(list_word)
    for word, count in counts.items():
        if count > 4:
            # sentence = sentence.replace(word, "")
            sentence = sentence.replace(word, "")
            sentence += word

    sentence = sentence.replace("  ", "")
    return sentence

def process(input):
    caption, description, info = input

    prompt = clean_text(caption)
    prompt = translater(prompt)
    prompt = gen_prompt(prompt)

    return prompt

