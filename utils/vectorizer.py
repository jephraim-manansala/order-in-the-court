import re
import json
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('wordnet')


TAX_STOPWORDS = []


def generate_stopwords(vocabs, documents):
    """Generate stopwords of a corpus"""

    # TfidfVectorizer from sklearn
    vectorizer = TfidfVectorizer(
        stop_words=None,
        norm=None,
        min_df=0,
        sublinear_tf=True,
        token_pattern=r'\b([a-z]+|\d{4,10})\b',
        vocabulary=vocabs
    )
    vectorizer.fit_transform(documents)
    # create dataframe from pandas
    tf_idf = pd.DataFrame({'word': vectorizer.vocabulary, 'idf':
                           vectorizer.idf_}, index=None)
    tf_idf.sort_values('idf', inplace=True, ascending=True)
    return tf_idf


def remove_noise(text, stop_words=TAX_STOPWORDS):
    """Tokenizer to be used for the TfidfVectorizer"""

    stop_words = stop_words or get_or_set_stopwords()
    wordnet_lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    cleaned_tokens = []

    for token in tokens:
        token = re.sub('[^A-Za-z]+', '', token)
        if len(token) > 1 and token.lower() not in stop_words:
            cleaned_tokens.append(wordnet_lemmatizer.lemmatize(token.lower()))

    return cleaned_tokens


def get_or_set_stopwords(stopwords=[]):
    """Get or set the module's stopwords"""

    global TAX_STOPWORDS
    if stopwords:
        TAX_STOPWORDS = stopwords
        return stopwords
    try:
        with open('utils/stopwords.json', 'r') as f:
            stopwords = json.load(f)
            TAX_STOPWORDS = stopwords
    except Exception as e:
        pass

    return stopwords or TAX_STOPWORDS


def get_vectorizer(docs):
    """
    Return tfidf vectorizer instance and the corresponding dataframe of
    the sparsed matrix
    """

    vect = None
    sparsed = None
    try:
        with open('utils/supremo.pkl', 'rb') as vf:
            vect = pickle.load(vf)
        with open('utils/supremo_sparsed.pkl', 'rb') as spf:
            sparsed = pickle.load(spf)
    except Exception as e:  # FileNotFoundError
        pass

    if not vect or sparsed is None:
        vect = TfidfVectorizer(tokenizer=remove_noise, use_idf=True,
                                smooth_idf=False, lowercase=True,
                                min_df=0.15, max_df=0.60)
        sparsed = vect.fit_transform(docs)
    vect_df = pd.DataFrame(columns=vect.get_feature_names(),
                           data=sparsed.toarray())
    return vect, vect_df

def vectorize_doc(vectorizer, doc):
    """Return a vectorized matrix of a doc using the specified vectorizer"""

    return vectorizer.transform(doc)
