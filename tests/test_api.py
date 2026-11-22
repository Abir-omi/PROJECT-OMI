"""API tests. A small model is trained on part of the data so the tests run fast."""

import os

import joblib
import pytest
from fastapi.testclient import TestClient

from bnc.data import download, load_split
from bnc.models import build


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    download()
    texts, labels = load_split("train")
    model = build("logistic_regression").fit(texts[:3000], labels[:3000])
    path = tmp_path_factory.mktemp("model") / "small.joblib"
    joblib.dump({"model": model, "name": "logistic_regression", "test": {"test_accuracy": 0.0, "test_macro_f1": 0.0}}, path)
    os.environ["MODEL_PATH"] = str(path)

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_sports(client):
    r = client.post("/predict", json={"text": "ক্রিকেট ম্যাচে সেঞ্চুরি করলেন অধিনায়ক, উইকেট নিলেন বোলার, সিরিজ জিতল দল"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "sports"
    assert body["label_bn"] == "খেলা"
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01
    assert list(body["probabilities"].values()) == sorted(body["probabilities"].values(), reverse=True)
    assert len(body["keywords"]) > 0


def test_predict_rejects_non_bangla(client):
    r = client.post("/predict", json={"text": "this is only english text"})
    assert r.status_code == 422


def test_predict_rejects_too_short(client):
    assert client.post("/predict", json={"text": "খেলা"}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"texts": ["নতুন ছবিতে অভিনয় করছেন নায়িকা, মুক্তি পাবে পুজোয়", "ফুটবল ম্যাচে গোল করে জিতল দল"]})
    assert r.status_code == 200
    assert [p["label"] for p in r.json()] == ["entertainment", "sports"]


def test_model_info(client):
    info = client.get("/model").json()
    assert set(info["classes"]) == {"entertainment", "international", "kolkata", "national", "sports", "state"}
