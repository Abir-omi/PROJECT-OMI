"""Text cleaning for Bangla news."""

import re
import unicodedata

# Common Bangla function words that carry little topic information
STOPWORDS = set(
    """
    এবং ও আর কিন্তু যে এই সেই ওই এক একটি একটা করে করা করেন করেছেন হয় হয়ে হয়েছে হয়েছিল
    ছিল ছিলেন থেকে জন্য সঙ্গে সাথে নিয়ে দিয়ে তার তাঁর তাদের তাঁদের আমি আমরা তিনি তারা
    কোনও কোন কি কী না নয় নেই বলে বলেন বলেছেন এখন তবে যদি বা হল হলো হবে হচ্ছে গিয়ে
    এর এটা এটি সব সবাই প্রায় অনেক কিছু আছে আছেন মধ্যে পরে আগে উপর দিকে কাছে মতো পর
    """.split()
)

_NON_BANGLA = re.compile(r"[^ঀ-৿\s]")
_BANGLA_DIGITS = re.compile(r"[০-৯]+")
_SPACES = re.compile(r"\s+")


def clean(text: str) -> str:
    """Normalise Unicode, keep only Bangla letters, replace numbers with a token, drop stopwords."""
    text = unicodedata.normalize("NFC", text)
    text = _NON_BANGLA.sub(" ", text)
    text = _BANGLA_DIGITS.sub(" ০ ", text)
    words = [w for w in _SPACES.split(text) if w and w not in STOPWORDS]
    return " ".join(words)
