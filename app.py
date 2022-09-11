import fastapi
import uvicorn
from _core.config import settings
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from tagger.routes import router as tagger
from reader.routes import router as reader
from chatbot.routes import router as chatbot
from counter.routes import router as counter
from creative.routes import router as creative
from recognizer.routes import router as recognizer


#from _database.services import create_database

def start_application():
    
    app = fastapi.FastAPI(
        title=settings.PROJECT_TITLE,
        description=settings.DESCRIPTION,
        version=settings.PROJECT_VERSION,
        openapi_tags=settings.TAGS,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    return app



templates = Jinja2Templates(directory = "templates")
app = start_application()
app.include_router(tagger)
app.include_router(reader)
app.include_router(chatbot)
app.include_router(counter)
app.include_router(creative)
app.include_router(recognizer)


if __name__ == "__main__":
    uvicorn.run(app)