from fastapi import APIRouter, UploadFile, File
from .services import *


router = APIRouter()


@router.post(
    "/apis/tagger/",
    tags=["tagger"])
async def image_label(image_file:UploadFile = File(...)):
    _object = await choose_class_async(image_file)

    return {"name":image_file.filename,
            "response":_object}
