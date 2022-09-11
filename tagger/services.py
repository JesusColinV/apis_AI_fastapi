from keras import models
import numpy as np
from PIL import Image
from fastapi import UploadFile

model = models.load_model('./tagger/IA/intel_image.h5')

async def choose_class_async(_image:UploadFile) -> str:
    image = Image.open(_image.file)
    image = image.resize((150,150))
    #image.save(os.path.join("./uploads", _image.filename))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1,150,150,3)

    # Predicting output
    result = model.predict(image_arr)
    ind = np.argmax(result)
    classes = ['buildings','forest','glacier','mountain','sea','street']
    return classes[ind]