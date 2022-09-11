
import numpy as np
from PIL import Image
from fastapi import UploadFile
import cv2
import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
             
async def read_text_async(_image:UploadFile) -> str:
    image = Image.open(_image.file)
    image_arr = np.array(image.convert('RGB'))
    gray_img_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(gray_img_arr)
    custom_config = r'-l eng --oem 3 --psm 6'
    text = pytesseract.image_to_string(image,config=custom_config)
    characters_to_remove = "!()@—*“>+-/,'|£#%$&^_~"
    new_string = text
    for character in characters_to_remove:
        new_string = new_string.replace(character, "")
    new_string = new_string.split("\n")
    return new_string
