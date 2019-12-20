import pickle
import pandas as pd

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))
finalized_model = pickle.load(open('models/finalized_model.pkl','rb'))
label_binarizer = pickle.load(open('models/label_binarizer.pkl','rb'))
medium_cleaned = pickle.load(open('models/medium_cleaned.pkl','rb'))
medium_tags = pickle.load(open('models/medium_tag_data.pkl','rb'))


def get_tag(cleaned_query):
    
    """
    Function which predicts the tag of the sentence
    based on the labeled tags given in the dataset    
    
    It first converts the processed sentence
    into a vector using TF-IDF vectorizer which 
    has been fitted on the whole corpus of words 
    in the Title column of Medium Stories.
    
    Logistic Regression  along with OneVsRestClassifier
    is used for predicting the label/tag
    
    The tags were converted to binary format using
    MultiLabelBinarizer. Using its inverse transform
    function, the tag name in string form is obtained.
    
    Parameters:
    cleaned_query (str): pre-processed input
    
    Returns:
    tag_name (list of strings): list of predicted tags
    
    Example: 
    cleaned query : "extract data from csv"
    tag_name: ['Tag-data-science']
    
    """
    # TF-IDF Vectorizer
    query_vector  = vectorizer.transform(pd.Series(cleaned_query))
    
    # Pre-trained Logistic Regression with OneVsRestClassifier
    query_pred = finalized_model.predict(query_vector)

    tag_name = label_binarizer.inverse_transform(query_pred)
    
    # inverse_transform function outputs list of tuples
    # following code converts it to list of strings
    tag_name = [list(x) for x in tag_name]
    tag_name = [item for sublist in tag_name for item in sublist]
    
    return tag_name