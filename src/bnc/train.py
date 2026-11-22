"""
Train and compare models, then save the best one.

    python -m bnc.train

Model selection uses the dev split only. The test split is touched once, at the end,
to report the final numbers honestly.
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from bnc.data import download, load_split  # noqa: E402
from bnc.models import MODEL_NAMES, build  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "news_classifier.joblib"
REPORTS = ROOT / "reports"


def main() -> None:
    download()
    x_train, y_train = load_split("train")
    x_dev, y_dev = load_split("dev")
    x_test, y_test = load_split("test")
    print(f"train={len(x_train)} dev={len(x_dev)} test={len(x_test)}")

    results = []
    best_name, best_f1, best_model = None, -1.0, None
    for name in MODEL_NAMES:
        start = time.time()
        model = build(name).fit(x_train, y_train)
        pred = model.predict(x_dev)
        acc, f1 = accuracy_score(y_dev, pred), f1_score(y_dev, pred, average="macro")
        secs = time.time() - start
        results.append({"model": name, "dev_accuracy": round(acc, 4), "dev_macro_f1": round(f1, 4), "train_seconds": round(secs, 1)})
        print(f"{name:22s} dev acc={acc:.4f} macro-F1={f1:.4f} ({secs:.0f}s)")
        if f1 > best_f1:
            best_name, best_f1, best_model = name, f1, model

    # Final, one-time evaluation of the chosen model on the untouched test set
    test_pred = best_model.predict(x_test)
    test = {
        "model": best_name,
        "test_accuracy": round(accuracy_score(y_test, test_pred), 4),
        "test_macro_f1": round(f1_score(y_test, test_pred, average="macro"), 4),
        "per_class": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
    }
    print(f"\nBest: {best_name}  test acc={test['test_accuracy']}  macro-F1={test['test_macro_f1']}")
    print(classification_report(y_test, test_pred, digits=3, zero_division=0))

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps({"dev_comparison": results, "test": test}, indent=2))
    save_plots(best_model, x_test, y_test, test_pred, results)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "name": best_name, "test": {k: test[k] for k in ("test_accuracy", "test_macro_f1")}}, MODEL_PATH, compress=3)
    print(f"Saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


def save_plots(model, x_test, y_test, test_pred, results) -> None:
    labels = list(model.classes_)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, labels=labels, normalize="true", values_format=".2f", cmap="PuRd", ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix (test set, row-normalised)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = [r["model"].replace("_", " ") for r in results]
    scores = [r["dev_macro_f1"] for r in results]
    colors = ["#f59e0b", "#06b6d4", "#8b5cf6", "#e11d48"]
    bars = ax.barh(names, scores, color=colors[: len(names)])
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set_xlim(min(scores) - 0.05, 1.0)
    ax.set_xlabel("Macro F1 on dev set")
    ax.set_title("Model comparison")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "model_comparison.png", dpi=150)
    plt.close(fig)

    counts = {c: int(np.sum(np.array(y_test) == c)) for c in labels}
    (REPORTS / "test_class_counts.json").write_text(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
