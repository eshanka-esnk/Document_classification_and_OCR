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
    dob = None
    adh = None
    sex = None
    add = None

    nameline = []
    dobline = []
    addline = []
    
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

    if 'female' in text.lower():
        sex = "FEMALE"
    else:
        sex = "MALE"
    
    text1 = list(filter(None, text1))
    text0 = text1[:]
    
    try:

        # Cleaning first names
        name = text0[0]
        name = name.rstrip()
        name = name.lstrip()
        name = name.replace("8", "B")
        name = name.replace("0", "D")
        name = name.replace("6", "G")
        name = name.replace("1", "I")
        name = re.sub('[^a-zA-Z] +', ' ', name)

        # Cleaning DOB
        dob = text0[1][-10:]
        dob = dob.rstrip()
        dob = dob.lstrip()
        dob = dob.replace('l', '/')
        dob = dob.replace('L', '/')
        dob = dob.replace('I', '/')
        dob = dob.replace('i', '/')
        dob = dob.replace('|', '/')
        dob = dob.replace('\"', '/1')
        dob = dob.replace(":","")
        dob = dob.replace(" ", "")

        # Cleaning Adhaar number details
        aadhar_number=''
        for word in res:
            if len(word) == 4 and word.isdigit():
                aadhar_number=aadhar_number  + word + ' '
        if len(aadhar_number)>=14:
            print("Aadhar number is :"+ aadhar_number)
        else:
            print("Aadhar number not read")
        adh=aadhar_number

        #cleaning address
        text0 = findword(text1, ('Address|Adress|ddress|Addess|Addrss|Addres|Add|Ad|Location)$'))
        addline = text0[0]
        add = addline.rstrip()
        add = add.lstrip()
        add = add.replace(" ", "")
        add = add.replace("\"", "")
        add = add.replace(";", "")
        add = add.replace("%", "L")
    except:
        pass
    
    
    
    data = {}
    data['Name'] = name
    data['DOB'] = dob
    data['Document Number'] = adh
    data['Sex'] = sex
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

# vOTER ID READ
def voterid_read_data(text):
    text1 = []
    
    full_name =""
    elder_name =""
    sex =""
    dob =""
    # Splitting the lines to sort the text paragraph wise
    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = s.rstrip()
        s = s.lstrip()
        text1.append(s)
        
    # Finding the electors number 
    voter_no = findword(text1, '(PASSPORT|CARD|IDENTITY CARD)$')
    voter_no = voter_no[0]
    voter_no = voter_no.replace(" ", "")
    
    lines = text
        
    for x in lines.split('\n'):
        _ = x.split()
        if ([w for w in _ if re.search("(Elector's|ELECTOR'S)$", w)]):    
            person_name = x
            person_name = person_name.split(':')[1].strip()
            full_name = person_name
                
        # Finding the father/husband/mother name        
        if ([w for w in _ if re.search("(Father's|Mother's|FATHER'S|MOTHER'S)$", w)]):
            elder_name = x
            elder_name = elder_name.split(':')[1].strip()
                
        # Finding the gender of the electoral candidate
        if ([w for w in _ if re.search('(Male|MALE|male)$', w)]):
            sex = "Male"
        elif ([w for w in _ if re.search('(Female|FEMALE|female)$', w)]):
            sex = "Female"
                
        # Finding the Date of Birth 
        if ([w for w in _ if re.search('(Year|YEAR|Birth|Date|Date of Birth|DATE OF BIRTH|DOB)$', w)]):
            dob = x
            dob = dob.split(':')[1].strip()
    
    # Converting the extracted informaton into json
    data = {
        'Document Number':voter_no,
        'Name':full_name,
        'Father Name':elder_name,
        'Sex':sex,
        'DOB':dob,
        'Document Type': "Voter ID"
    }
    return data
def findword(textlist, wordstring):
    lineno = -1
    for wordline in textlist:
        xx = wordline.split()
        if ([w for w in xx if re.search(wordstring, w)]):
            lineno = textlist.index(wordline)
            textlist = textlist[lineno+1:]
            return textlist
    return textlist
# DRIVERS LICENCE
def Driving_Licence_read(text):
    
    name = None
    add = None
    dob = None
    did = None
    
    nameline = []
    addline = []
    dobline = []
    didline =[]
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
            if ([w for w in xx if re.search('(DRIVING LICENSE|DRIVER|DRIVE|DIVE|RIVE|DRVE|DRIE|DRIV|TRANSPORT|STATE|DRI|IVE|LICENCE|LICEN)$', w)]):
                text1 = list(text1)
                lineno = text1.index(wordline)
                break
    text0 = text1[lineno+1:]
    try:
        #cleaning number
        text0 = findword(text1, '(Number|umber|Vehicle|N.|Vehi|Transport|Num)$')
        didline = text0[0]
        did = didline.rstrip()
        did = did.lstrip()
        did = did.replace(" ", "")
        did = did.replace("\"", "")
        did = did.replace(";", "")
        did = did.replace("%", "L")
        #cleaning name
        name = text0[0]
        name = name.rstrip()
        name = name.lstrip()
        name = name.replace("8", "B")
        name = name.replace("0", "D")
        name = name.replace("6", "G")
        name = name.replace("1", "I")
        name = re.sub('[^a-zA-Z] +', ' ', name)
        #cleaning address
        text0 = findword(text1, ('Address|Adress|ddress|Addess|Addrss|Addres|Add|Ad|Location)$'))
        addline = text0[0]
        add = addline.rstrip()
        add = add.lstrip()
        add = add.replace(" ", "")
        add = add.replace("\"", "")
        add = add.replace(";", "")
        add = add.replace("%", "L")
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
    except:
        pass
    data = {}
    data['Name'] = name
    data['Address'] = add
    data['Document Number'] = did
    data['DOB'] = dob
    data['Document Type'] = "Driving License"
    
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