from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse, Response
from starlette.responses import StreamingResponse
from .services import *
from PIL import Image

router = APIRouter()


@router.post("/detect/face",
             tags=["recognizer"])
async def predict(file: bytes = File(...)):
    _object = await detect_face(file)
    return {
        "count":len(_object),
        "face":_object
    }

@router.post("/detect/card",
             tags=["recognizer"])
async def predict_card_scoreNdiff(file: UploadFile = File(...)):
    (score, diff) = await predic_card(file)
    return {
        "score": score,
        "diff": len(diff)
    }
    

@router.post("/detect/card/image",
             tags=["recognizer"])
async def predic__card_show_image(file: UploadFile = File(...),op:str = "diff"):
    _object = await predic_card_image(file)
    return StreamingResponse(io.BytesIO(_object.tobytes()), media_type="image/png")
    #return {"Error":"problemas para concluir el desarrollo"}
    

@router.post("/detect/sentiment_analyzer",
             tags=["recognizer"])
async def predic_sentiment(text:str):
    score = await sentiment_analizer(text)
    opinion = lambda score : ["p" if score > 0.6 else "n"]
    return {
        "score":score,
        "opinion": opinion(score) 
    }
