from pydantic import BaseModel
#from PIL import Image


        
class IImage(BaseModel):
    image:bytes 
    count:int 
    
    class Config:
        orm_mode = True

class IImageBusNCar(BaseModel):
    image:bytes
    bus:int 
    car:int 
    
    class Config:
        orm_mode = True
