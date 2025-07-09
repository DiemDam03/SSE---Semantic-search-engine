import re
import unicodedata

# preprocess
def preprocessor(text):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() 