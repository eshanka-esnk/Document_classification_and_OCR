from venv import logger
from pdf2image import convert_from_path
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pytesseract
from PIL import Image
import ftfy
import io
import json
import sys

def pdf2img(file_path: str):
    try:
        images = convert_from_path(file_path)
        for i, img in enumerate(images):
            img.save(f'extracted\image{i}.jpg', 'JPEG')
    except Exception as e:
        logger.error(e)

def to_unicode(data):
    if sys.version_info[0] < 3:  # Python 2
        return unicode(data, "utf-8") if isinstance(data, str) else data
    return data

def extract_info(filename: str, label: str):
    img = cv2.imread(filename)
    try:
        img = cv2.resize(img, None, fx=2, fy=2,interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.inRange(img, 0, 75)
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        print(img.shape)
    except:
        img=img
        print('none')


    kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
    img = cv2.filter2D(src=img, depth=-1, kernel=kernel)

    text = pytesseract.image_to_string(Image.open(filename), lang = 'eng')
    #text = pytesseract.image_to_string(img, lang = 'eng')

    text_output = open('output.txt', 'w', encoding='utf-8')
    text_output.write(text)
    text_output.close()

    file = open('output.txt', 'r', encoding='utf-8')
    text = file.read()
    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)

    #label = "Adhaar Card"|"PAN"|"Voter ID Card"|"Driving License"
    if label == "PAN":
        data = pan_read_data(text)
    elif label == "Adhaar Card":
        data = adhaar_read_data(text)
    elif label == "Voter ID Card":
        data = voterid_read_data(text)
    elif label == "Driving License":
        data = Driving_Licence_read(text)

    with io.open('info.json', 'w', encoding='utf-8') as outfile:
        data = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '))
        #data = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(data))
    with open('info.json', encoding = 'utf-8') as data:
        data_loaded = json.load(data)
    return data_loaded

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break