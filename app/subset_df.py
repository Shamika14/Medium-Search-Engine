import pickle
medium_tags = pickle.load(open('models/medium_tag_data.pkl','rb'))
medium_cleaned = pickle.load(open('models/medium_cleaned.pkl','rb'))

def subset_df(tag_name):
    
    """
    Function that subsets the dataset by filtering 
    on the basis of tags
    
    It collects the index values of observations 
    with the predicted tag name and subsets the
    data to those values
    
    This reduces the corpus to only the related observations,
    reducing the computation time.
    
    Parameters:
    tag_name (list): list of tags/labels
    
    Returns:
    search_subset_df (DataFrame) : subset of dataset with
                                    relevant tags
    
    """
    ind_subset = [] 
    
    # medium_tags is a pandas Series with represents the tag_names
    for row in range(medium_tags.shape[0]):
        for column in tag_name:
            if column in medium_tags[row]:
                ind_subset.append(row)
        
    #medium_cleaned is the whole dataframe without tags column
    search_subset_df = medium_cleaned[['Title','Claps','url']].iloc[ind_subset]
    
    return search_subset_df
