import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation


words = set(nltk.corpus.words.words())
stuff_to_be_removed = list(stopwords.words("english"))+list(punctuation)

lemmatizer = WordNetLemmatizer()

def clean_text(sent):
    
    """
    Function to process search query text
    
    Converts text to lowercase, removes stop words
    and punctuation and uses WordNetLemmatizer to get 
    root words.
    
    Parameters:
    sent(str): input sentence
    
    Returns:
    cleaned_sent (str): processed string
    """
    
    text = word_tokenize(sent.lower())
    text = [t for t in text if len(t) > 2]
    text = [lemmatizer.lemmatize(y) for y in text if y not in stuff_to_be_removed]
    cleaned_sent = " ".join(text)
    return cleaned_sent 