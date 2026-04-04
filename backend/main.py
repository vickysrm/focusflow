from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

load_dotenv()

from routers import session, transcript, qa
from ml.embeddings import load_embedding_model
from ml.topic_model import load_topic_model
from ml.classifier import load_classifier
from ml.drift_detector import load_drift_model
from ml.rewriter import load_rewriter

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML models...")
    app.state.embedder = load_embedding_model()
    app.state.topic_model = load_topic_model()
    app.state.classifier = load_classifier()
    app.state.drift_model = load_drift_model()
    app.state.rewriter = load_rewriter()
    print("All ML models loaded.")
    yield
    print("Shutting down.")

app = FastAPI(title="FocusFlow API", version="1.0.0", lifespan=lifespan)

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router, prefix="/session", tags=["session"])
app.include_router(transcript.router, prefix="/transcript", tags=["transcript"])
app.include_router(qa.router, prefix="/qa", tags=["qa"])

from auth import get_current_user
from fastapi import Depends

@app.get("/health", dependencies=[Depends(get_current_user)])
def health():
    return {"status": "ok", "service": "FocusFlow API"}