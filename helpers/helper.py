import csv
import re
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

#### Reused code from git repo provided

#ADHAAR READ
def adhaar_read_data(text):
    res=text.split()
    name = None
    yob = None
    gender = None
    adhar = None
    nameline = []
    dobline = []
    panline = []
    text0 = []
    text1 = []
    text2 = []

    # Searching for PAN
    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = lin.replace('\n','')
        s = s.rstrip()
        s = s.lstrip()
        text1.append(s)

    text1 = list(filter(None, text1))
    # print(text1)

    # to remove any text read from the image file which lies before the line 'Income Tax Department'

    lineno = 0  # to start from the first line of the text file.

    for wordline in text1:
        xx = wordline.split('\n')
        if ([w for w in xx if re.search('(INCOMETAXDEPARWENT @|mcommx|INCOME|TAX|GOW|GOVT|GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT|PARTMENT|ARTMENT|INDIA|NDIA)$', w)]):
            text1 = list(text1)
            lineno = text1.index(wordline)
            break

    # text1 = list(text1)
    text0 = text1[lineno+1:]
    print(text0)  # Contains all the relevant extracted text in form of a list - uncomment to check

    def findword(textlist, wordstring):
        lineno = -1
        for wordline in textlist:
            xx = wordline.split( )
            if ([w for w in xx if re.search(wordstring, w)]):
                lineno = textlist.index(wordline)
                textlist = textlist[lineno+1:]
                return textlist
        return textlist

    ###############################################################################################################
    ######################################### Section 5: Dishwasher part ##########################################
    ###############################################################################################################

    try:

        # Cleaning Name
        name = text0[2]
        name = name.rstrip()
        name = name.lstrip()
        name = re.sub('[^a-zA-Z] +', ' ', name)

        # Cleaning YOB
        yob = text0[3]
        yob = re.sub('[^0-9]+', '', yob)
        yob = yob.replace(" ", "")
        yob = yob[2:6]
        yob = yob.rstrip()
        yob = yob.lstrip()

        # Cleaning Gender
        gender = text0[4]
        gender = gender.replace('/', '')
        gender = gender.replace('(', '')
        gender = gender.rstrip()
        gender = gender.lstrip()

        # Cleaning Aadhar Number
        adhar = text0[5]
        adhar = adhar.rstrip()
        adhar = adhar.lstrip()

    except:
        pass

    # Making tuples of data
    data = {}
    data['Name'] = name
    data['Year of Birth'] = yob
    data['Gender'] = gender
    data['Number'] = adhar
    return data

def findword(textlist, wordstring):
    lineno = -1
    for wordline in textlist:
        xx = wordline.split( )
        if ([w for w in xx if re.search(wordstring, w)]):
            lineno = textlist.index(wordline)
            textlist = textlist[lineno+1:]
            return textlist
    return textlist

#PAN READ
def pan_read_data(text):
    name = None
    fname = None
    dob = None
    pan = None
    nameline = []
    dobline = []
    panline = []
    text0 = []
    text1 = []
    text2 = []
    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = lin.replace('\n','')
        s = s.rstrip()
        s = s.lstrip()
        text1.append(s)
    text1 = list(filter(None, text1))
    lineno = 0
    for wordline in text1:
            xx = wordline.split('\n')
            if ([w for w in xx if re.search('(INCOMETAXDEPARWENT|INCOME|TAX|GOW|GOVT|GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT|PARTMENT|ARTMENT|INDIA|NDIA)$', w)]):
                text1 = list(text1)
                lineno = text1.index(wordline)
                break
    text0 = text1[lineno+1:]
    try:
            name = text0[0]
            name = name.rstrip()
            name = name.lstrip()
            name = name.replace("8", "B")
            name = name.replace("0", "D")
            name = name.replace("6", "G")
            name = name.replace("1", "I")
            name = re.sub('[^a-zA-Z] +', ' ', name)
            # Cleaning Father's name
            fname = text0[1]
            fname = fname.rstrip()
            fname = fname.lstrip()
            fname = fname.replace("8", "S")
            fname = fname.replace("0", "O")
            fname = fname.replace("6", "G")
            fname = fname.replace("1", "I")
            fname = fname.replace("\"", "A")
            fname = re.sub('[^a-zA-Z] +', ' ', fname)
            # Cleaning DOB
            dob = text0[2][:10]
            dob = dob.rstrip()
            dob = dob.lstrip()
            dob = dob.replace('l', '/')
            dob = dob.replace('L', '/')
            dob = dob.replace('I', '/')
            dob = dob.replace('i', '/')
            dob = dob.replace('|', '/')
            dob = dob.replace('\"', '/1')
            dob = dob.replace(" ", "")
            # Cleaning PAN Card details
            text0 = findword(text1, '(Pormanam|Number|umber|Account|ccount|count|Permanent|ermanent|manent|wumm)$')
            panline = text0[0]
            pan = panline.rstrip()
            pan = pan.lstrip()
            pan = pan.replace(" ", "")
            pan = pan.replace("\"", "")
            pan = pan.replace(";", "")
            pan = pan.replace("%", "L")
    except:
            pass
    data = {}
    data['Name'] = name
    data['Father Name'] = fname
    data['DOB'] = dob
    data['Document Number'] = pan
    data['Document Type'] = "PAN"
    return data

def findword(textlist, wordstring):
    lineno = -1
    for wordline in textlist:
        xx = wordline.split( )
        if ([w for w in xx if re.search(wordstring, w)]):
            lineno = textlist.index(wordline)
            textlist = textlist[lineno+1:]
            return textlist
    return textlist

# Passport READ
def passport_read_data(text):
    surname = None
    first_name = None
    dob = None
    gender = None
    number = None
    doe = None
    text0 = []
    text1 = []

    # Searching for PAN
    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = lin.replace('\n','')
        s = s.rstrip()
        s = s.lstrip()
        text1.append(s)

    text1 = list(filter(None, text1))
    # print(text1)

    # to remove any text read from the image file which lies before the line 'Income Tax Department'

    lineno = 0  # to start from the first line of the text file.

    # text1 = list(text1)
    text0 = text1[lineno+1:]
    print(text0)  # Contains all the relevant extracted text in form of a list - uncomment to check

    ###############################################################################################################
    ######################################### Section 5: Dishwasher part ##########################################
    ###############################################################################################################
    try:

        # Cleaning Surname
        surname = text0[3]
        surname = surname.rstrip()
        surname = surname.lstrip()
        surname = re.sub('[^a-zA-Z] +', ' ', surname)

        # Cleaning First Name
        first_name = text0[5]
        first_name = first_name.rstrip()
        first_name = first_name.lstrip()
        first_name = re.sub('[^a-zA-Z] +', ' ', first_name)

        # Cleaning DOB
        dob = text0[7]
        dob = dob.rstrip()
        dob = dob.lstrip()
        dob = dob[-12:]

        # Cleaning Gender
        gender = text0[4]
        gender = 'M' # need to fix this

        # Cleaning Passport Number
        number = text0[1]
        number = number[-8:]
        number = number.rstrip()
        number = number.lstrip()

        # Cleaning DOE
        doe = text0[14]
        doe = doe.rstrip()
        doe = doe.lstrip()
        doe = doe[-12:-2]

    except:
        pass

    # Making tuples of data
    data = {}
    data['Surname'] = surname
    data['First Name'] = first_name
    data['Date of Birth'] = dob
    data['Gender'] = gender
    data['Number'] = number
    data['Date of Expiry'] = doe
    data['Document Type'] = "Passport"
    return data

# DRIVERS LICENCE
def Driving_Licence_read(text):
    
    name = None
    surname = None
    dob = None
    doe = None
    number = None
    text0 = []
    text1 = []

    # Searching for PAN
    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = lin.replace('\n','')
        s = s.rstrip()
        s = s.lstrip()
        text1.append(s)

    text1 = list(filter(None, text1))
    # print(text1)

    # to remove any text read from the image file which lies before the line 'Income Tax Department'

    lineno = 0  # to start from the first line of the text file.

    # text1 = list(text1)
    text0 = text1[lineno+1:]
    print(text0)  # Contains all the relevant extracted text in form of a list - uncomment to check

    ###############################################################################################################
    ######################################### Section 5: Dishwasher part ##########################################
    ###############################################################################################################
    try:

        # Cleaning name
        name = text0[5]
        name = re.sub('[^a-zA-Z]+', ' ', name)
        name = name.rstrip()
        name = name.lstrip()

        # Cleaning DOB
        dob = text0[4]
        dob = dob.rstrip()
        dob = dob.lstrip()
        dob = dob[4:7]

        # Cleaning Passport Number
        number = text0[1]
        number = number[-21:-6]
        number = number.rstrip()
        number = number.lstrip()

        # Cleaning DOE
        doe = text0[14]
        doe = doe.rstrip()
        doe = doe.lstrip()
        doe = doe[-12:-2]

    except:
        pass

    # Making tuples of data
    data = {}
    data['Name'] = name
    data['Date of Birth'] = dob
    data['Number'] = number
    data['Date of Expiry'] = doe
    data['Document Type'] = "Driving License"
    return data
    
def findword(textlist, wordstring):
    lineno = -1
    for wordline in textlist:
        xx = wordline.split( )
        if ([w for w in xx if re.search(wordstring, w)]):
            lineno = textlist.index(wordline)
            textlist = textlist[lineno+1:]
            return textlist
    return textlist

#####

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
        data = passport_read_data(text)
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
    
def save_to_csv(jsondata):
    # jsondata = json.loads(data)
    data_file = open('extracted/output.csv', 'w', newline='')
    csv_writer = csv.writer(data_file)
    count = 0
    if count == 0:
        header = jsondata.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(jsondata.values())
    
    data_file.close()