import pickle

from clean_text import clean_text
from get_tag import get_tag
from subset_df import subset_df
from search_result import search_result

medium_cleaned = pickle.load(open('models/medium_cleaned.pkl','rb'))

def get_search_result(query):
    
        """ Main Program """
        
        cleaned_query = clean_text(query)
        
        tag_name = get_tag(cleaned_query)
        # If tag name is None, the whole Title column
        # is given as input to search result to 
        # find similarity with the entire corpus
        if tag_name:           
            df = subset_df(tag_name)        
            output_var = search_result(df,cleaned_query)
        else:
            output_var = search_result(medium_cleaned,cleaned_query)
            
        return output_var

''' Following code was for testing this function without html '''
'''
if __name__ == '__main__':
    print("Testing if API works")
    print('input string is ')
    query = 'how to extract data from csv'
    print(query)
    result = get_search_result(query)
    print(f'Input values: {query}')
    print(f'Output URL: {result}')
'''