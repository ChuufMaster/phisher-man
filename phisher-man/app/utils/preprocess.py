import re
import warnings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from difflib import SequenceMatcher
import jellyfish

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def contains_url(text):
    if pd.isna(text):
        return 0
    # Simple URL pattern
    url_pattern = re.compile(r'https?://|www\.\S+|ftp://|\S+\.(com|net|org|info|io|biz|co|me|ly)\b')
    return int(bool(url_pattern.search(text)))

def count_urls(text):
    url_pattern = re.compile(r'https?://|www\.\S+|ftp://|\S+\.(com|net|org|info|io|biz|co|me|ly)\b')
    return len(url_pattern.findall(text or ""))

def similar(a, b):
    if a == "unknown" == b:
        return 0
    # return 1.0 - SequenceMatcher(None, a, b).ratio()
    return jellyfish.levenshtein_distance(a, b)

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = str(text).lower()
    text = re.sub(r"http\S+|www|S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text.strip()

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words])

