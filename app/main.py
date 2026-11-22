import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bnc.predict import DEFAULT_MODEL, NewsClassifier

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "static"
classifier: NewsClassifier | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global classifier
    classifier = NewsClassifier(Path(os.getenv("MODEL_PATH", DEFAULT_MODEL)))
    yield


app = FastAPI(
    title="Bangla News Classifier",
    description="Classifies a Bangla news article into one of six topics and explains which words drove the decision.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/reports", StaticFiles(directory=ROOT / "reports"), name="reports")


class ArticleIn(BaseModel):
    text: str = Field(min_length=10, max_length=20_000, examples=["বিশ্বকাপ ক্রিকেটে দারুণ সেঞ্চুরি করে দলকে জেতালেন অধিনায়ক"])


class BatchIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=50)


class PredictionOut(BaseModel):
    label: str
    label_bn: str
    confidence: float
    probabilities: dict[str, float]
    keywords: list[str]


def model() -> NewsClassifier:
    if classifier is None:
        raise HTTPException(503, "Model not loaded")
    return classifier


@app.post("/predict", response_model=PredictionOut, tags=["Model"])
def predict(body: ArticleIn):
    try:
        return model().predict(body.text).__dict__
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/predict/batch", response_model=list[PredictionOut], tags=["Model"])
def predict_batch(body: BatchIn):
    out = []
    for text in body.texts:
        try:
            out.append(model().predict(text).__dict__)
        except ValueError as e:
            raise HTTPException(422, f"{e}: {text[:40]}")
    return out


@app.get("/model", tags=["Model"])
def model_info():
    """Which model is running and how well it scored on the held-out test set."""
    metrics_file = ROOT / "reports" / "metrics.json"
    comparison = json.loads(metrics_file.read_text())["dev_comparison"] if metrics_file.exists() else []
    return {"model": model().name, "classes": model().classes, "test": model().metrics, "comparison": comparison}


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(STATIC / "index.html")
