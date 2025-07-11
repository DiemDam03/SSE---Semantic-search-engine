import re
import unicodedata

# preprocess
def preprocessor(text):
    text = unicodedata.normalize("NFC", text) # normalize unicode
    text = text.lower() # convert to lowercase
    text = re.sub(r"[^\w\s]", "", text) # remove punctuation
    # text = re.sub(r"<[^>]+>", "", text) # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text.split() 