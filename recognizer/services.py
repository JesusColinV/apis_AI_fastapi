

import cv2
import io
import numpy as np
import os
from fastapi import UploadFile
from skimage.metrics import structural_similarity
from PIL import Image
#from sentiment_analysis_spanish import sentiment_analysis


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
face_cascade_src = base_dir+r'\recognizer\AI\haarcascade_frontalface_default.xml'
reference = base_dir+r'\recognizer\reference\image.jpg'
#sentiment = sentiment_analysis.SentimentAnalysisSpanish()

async def read_cv2_image(binaryimg):

    stream = io.BytesIO(binaryimg)
    image = np.asarray(bytearray(stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

async def detect_face(binaryimg:UploadFile) -> list: 

    # convert the binary image to image
    image = await read_cv2_image(binaryimg)

    # convert the image to grayscale
    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load the face cascade detector,
    facecascade = cv2.CascadeClassifier(face_cascade_src)

    # detect faces in the image
    facedetects = facecascade.detectMultiScale(imagegray, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # construct a list of bounding boxes from the detection
    facerect = [(int(fx), int(fy), int(fx + fw), int(fy + fh)) for (fx, fy, fw, fh) in facedetects]

    # update the data dictionary with the faces detected
    return facerect


async def predic_card(_image_uploaded:UploadFile) -> str:

    # Read uploaded and original image as array
    #original_image = cv2.imread(_image_original.file)
    original_image = cv2.imread(reference)
    #original_image = Image.open(os.path.join(reference, 'image.jpg')).resize((250,160))
    image = Image.open(_image_uploaded.file)
    uploaded_image = np.array(image.convert('RGB'))
    #uploaded_image = cv2.imread(_image_uploaded)

    # Convert image into grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

    # Calculate structural similarity
    (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
    
    return (score,diff)

async def sentiment_analizer(data:str) ->float:
    #score = sentiment.sentiment(data)
    score = 10
    return score


async def predic_card_image(_image_uploaded:UploadFile, op:str = "diff"):
    # Read uploaded and original image as array
    original_image = cv2.imread(reference)
    #original_image = cv2.imread(_image_original.file)
    image = Image.open(_image_uploaded.file)
    uploaded_image = np.array(image)
    # Convert image into grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    # Calculate structural similarity
    (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
    diff = (diff * 255).astype("uint8")
    if op == "diff":
        diff = Image.fromarray(diff)
        res, im_png = cv2.imencode(".png", np.array(diff))
        return im_png
    else:
        # Calculate threshold and contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = Image.fromarray(thresh)
        res, im_png = cv2.imencode(".png", np.array(thresh))
        return im_png
    