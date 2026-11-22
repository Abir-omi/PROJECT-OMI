"""Download and load the Bangla news categorization dataset."""

import re
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BASE_URL = "https://raw.githubusercontent.com/banglanlp/bnlp-resources/main/news_categorization"
SPLITS = ("train", "dev", "test")

LABELS_BN = {
    "kolkata": "কলকাতা",
    "state": "রাজ্য",
    "national": "জাতীয়",
    "international": "আন্তর্জাতিক",
    "sports": "খেলা",
    "entertainment": "বিনোদন",
}


def download(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = data_dir / f"{split}.tsv"
        if not target.exists():
            print(f"Downloading {split}.tsv ...")
            urllib.request.urlretrieve(f"{BASE_URL}/{split}.tsv", target)


def load_split(split: str, data_dir: Path = DATA_DIR) -> tuple[list[str], list[str]]:
    """
    Read one split. The file is tab separated (id, text, label), but some articles
    contain line breaks or tabs, so a normal CSV reader breaks on them. A new row
    always starts with "<number><tab>", so lines that don't are joined to the row above.
    """
    path = data_dir / f"{split}.tsv"
    rows: list[str] = []
    with path.open(encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if re.match(r"^\d+\t", line) or not rows:
                rows.append(line)
            else:
                rows[-1] += " " + line

    texts, labels = [], []
    for row in rows:
        _, rest = row.split("\t", 1)
        text, label = rest.rsplit("\t", 1)
        texts.append(text.replace("\t", " "))
        labels.append(label.strip())
    return texts, labels
