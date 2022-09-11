
import numpy as np
from PIL import Image
from fastapi import UploadFile
import cv2
import numpy as np
from PIL import Image
import os
from .schema import *
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
car_cascade_src = base_dir+r'\counter\AI\cars.xml'
bus_cascade_src = base_dir+r'\counter\AI\Bus_front.xml'

async def identify_car_async(_image:UploadFile) -> IImage:
    image = Image.open(_image.file)
    image = image.resize((450,250))
    image_arr = np.array(image)
    grey = cv2.cvtColor(image_arr,cv2.COLOR_BGR2GRAY)
    ccnt = 0
    
    #Cascade
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    cars = car_cascade.detectMultiScale(grey, 1.1, 1)
    
    for (x,y,w,h) in cars:
            cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
            ccnt += 1
    res, im_png = cv2.imencode(".png", image_arr)
    response = IImage(image = im_png.tobytes(), count = ccnt)
    return response
    #img = Image.fromarray(image_arr, 'RGB')
    #response = IImage(image = img.tobytes(), count = ccnt)
    #return response

async def identify_bus_async(_image:UploadFile) -> IImage:
    image = Image.open(_image.file)
    image = image.resize((450,250))
    image_arr = np.array(image)
    grey = cv2.cvtColor(image_arr,cv2.COLOR_BGR2GRAY)
    bcnt = 0
    
    bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
    bus = bus_cascade.detectMultiScale(grey, 1.1, 1)
    
    for (x,y,w,h) in bus:
        cv2.rectangle(image_arr,(x,y),(x+w,y+h),(0,255,0),2)
        bcnt += 1
    
    #img = Image.fromarray(image_arr, 'RGB')
    res, im_png = cv2.imencode(".png", image_arr)
    response = IImage(image = im_png.tobytes(), count = bcnt)
    return response
    #response = IImage(image = im_png.tobytes(), bus = bcnt, car = ccnt)
    #return response

async def identify_both_async(_image:UploadFile) -> IImageBusNCar:
    image = Image.open(_image.file)
    image = image.resize((450,250))
    image_arr = np.array(image)
    grey = cv2.cvtColor(image_arr,cv2.COLOR_BGR2GRAY)
    bcnt = 0
    ccnt = 0
    
    #Cascade
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    cars = car_cascade.detectMultiScale(grey, 1.1, 1)

    
    bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
    bus = bus_cascade.detectMultiScale(grey, 1.1, 1)
    
    for (x,y,w,h) in bus:
        cv2.rectangle(image_arr,(x,y),(x+w,y+h),(0,255,0),2)
        bcnt += 1
        
    if bcnt == 0:
        for (x,y,w,h) in cars:
            cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
            ccnt += 1
    res, im_png = cv2.imencode(".png", image_arr)
    #img = Image.fromarray(image_arr, 'RGB')
    response = IImageBusNCar(image = im_png.tobytes(), bus = bcnt, car = ccnt)
    return response
    #res, im_png = cv2.imencode(".png", result)
    #return im_png