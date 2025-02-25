from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def advanced_preprocess(text):
    words = simple_preprocess(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words
