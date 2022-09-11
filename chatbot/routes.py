from fastapi import APIRouter
from .services import *


router = APIRouter()


@router.post(
    "/apis/chatbot/",
    tags=["bot"])
def bot_response(input:str):
    _object = bot_response_async()
    return {"response":_object}
