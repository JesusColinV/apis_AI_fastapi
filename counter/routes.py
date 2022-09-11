from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse, Response
from starlette.responses import StreamingResponse
from .services import *
from .schema import *
import io
from PIL import Image

router = APIRouter()


@router.post(
    "/apis/counter/car/image",
    tags=["counter"])
async def image_car(image_file:UploadFile = File(...)):
    _object:IImage = await identify_car_async(image_file)
    imgio = io.BytesIO(_object.image)
    imgio.seek(0)
    #return FileResponse(_object.image)
    #return Response(content=imgio, media_type="image/png")
    return StreamingResponse(content=imgio, media_type="image/jpeg")
  
@router.post(
    "/apis/counter/car/count",
    tags=["counter"])
async def count_car(image_file:UploadFile = File(...)):
    _object:IImage = await identify_car_async(image_file)
    return {
        "count":_object.count
    }


@router.post(
    "/apis/counter/bus/image",
    tags=["counter"])
async def image_bus(image_file:UploadFile = File(...)):
    _object:IImage = await identify_bus_async(image_file)
    return StreamingResponse(io.BytesIO(_object.image), media_type="image/png")

@router.post(
    "/apis/counter/bus/count",
    tags=["counter"])
async def count_bus(image_file:UploadFile = File(...)):
    _object:IImage = await identify_bus_async(image_file)
    return {
        "count":_object.count
    }

    
@router.post(
    "/apis/counter/transport/image",
    tags=["counter"])
async def images_all(image_file:UploadFile = File(...)):
    _object:IImageBusNCar = await identify_both_async(image_file)
    return StreamingResponse(io.BytesIO(_object.image), media_type="image/png")

@router.post(
    "/apis/counter/transport/count",
    tags=["counter"])
async def count_all(image_file:UploadFile = File(...)):
    _object:IImageBusNCar = await identify_both_async(image_file)
    return {
        "count_bus":_object.bus,
        "count_car":_object.car
    }
