from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import FileResponse, Response
from starlette.responses import StreamingResponse
from .services import *
import io
from PIL import Image

router = APIRouter()


@router.post(
    "/apis/creative/watermaking/image",
    tags=["creative"]
    )
async def image_watermakinr(image1_file:UploadFile = File(...),image2_file:UploadFile = File(...)):
    _object = await watermarking_async(image1_file,image2_file)
    return StreamingResponse(io.BytesIO(_object.tobytes()), media_type="image/png")
    #return FileResponse(_object.save("gee.jpeg"),media_type="image/jpeg")

#StreamingResponse(content=_object, media_type="image/jpeg")
    
@router.post(
    "/apis/creative/textmark/image",
    tags=["creative"])
async def image_textmark(image1_file:UploadFile = File(...),text:str = "Javithon equipo ganador"):
    _object = await textmark_async(image1_file,text)
    return StreamingResponse(io.BytesIO(_object.tobytes()), media_type="image/png")

@router.post(
    "/apis/creative/face_swap/image",
    tags=["creative"])
async def image_face_swap(face1:UploadFile,face2:UploadFile):
    _object = await face_swap_async(face1,face2)
    return StreamingResponse(io.BytesIO(_object.tobytes()), media_type="image/png")

