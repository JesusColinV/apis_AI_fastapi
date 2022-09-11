from fastapi import APIRouter, UploadFile, File
from .services import *


router = APIRouter()


@router.post(
    "/apis/reader/",
    tags=["reader"])
async def image_text(image_file:UploadFile = File(...)):
    _object = await read_text_async(image_file)
    return {"name":image_file.filename,
            "response":_object}

