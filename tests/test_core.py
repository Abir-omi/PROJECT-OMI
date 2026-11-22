from bnc.data import load_split
from bnc.preprocess import clean


def test_clean_keeps_bangla_and_drops_noise():
    out = clean("ঢাকায় ১২৩ জন ও Hello! খেলা দেখতে এসেছেন।")
    assert "Hello" not in out
    assert "১২৩" not in out and "০" in out  # numbers become one token
    assert "ও" not in out.split()  # stopword removed
    assert "খেলা" in out


def test_clean_empty_for_non_bangla():
    assert clean("only english text 123") == ""


def test_loader_joins_broken_lines(tmp_path):
    (tmp_path / "train.tsv").write_text(
        "id\ttext\tclass_label\n"
        "0\tপ্রথম খবর\tsports\n"
        "1\tদ্বিতীয় খবরের প্রথম লাইন\n"
        "দ্বিতীয় লাইন\twith a tab\tnational\n"
        "2\tতৃতীয় খবর\tstate\n",
        encoding="utf-8",
    )
    texts, labels = load_split("train", tmp_path)
    assert labels == ["sports", "national", "state"]
    assert "দ্বিতীয় লাইন" in texts[1]
    assert "\t" not in texts[1]
