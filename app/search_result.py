import numpy as np
from gensim.models import Word2Vec
from scipy import spatial


def search_result(search_subset_df,cleaned_query):
    
    """
    Function that calculates cosine similarity
    between the input search query and the dataset Titles
    and outputs the URL with the highest similarity value
    and highest claps.
    
    It updates Word2Vec with vocab of the subset corpus
    and average word embedding for each of the sentences is
    calculated and their cosine similarity scores computed

    The output URL is that observation with highest similarity
    score and highest claps received.
    
    Parameters:
    cleaned_query (str): pre-processed input
    search_subset_df (DataFrame): subset of dataset with
                                    relevant tags
    
    Returns:
    output_url (str): best possible output in the form of URL
        
    """    
    corpus_subset = search_subset_df['Title'].apply(lambda x : x.split())
    corpus_subset = corpus_subset.tolist()
        
    model_w2c = Word2Vec(size=100, min_count=1)
    model_w2c.build_vocab(sentences=corpus_subset)
    index2word_set = set(model_w2c.wv.index2word)
    
    s1_afv = avg_feature_vector(cleaned_query,model=model_w2c,num_features=100,index2word_set=index2word_set)
    sim_score = []

    for row in range(search_subset_df.shape[0]):
    
        s2_afv = avg_feature_vector(search_subset_df['Title'].iloc[row],model=model_w2c,num_features=100,index2word_set=index2word_set)    
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        sim_score.append(sim)
    
    search_subset_df['sim_score'] = sim_score 
    
    output_df = search_subset_df.sort_values(by=['sim_score','Claps'],ascending=False)
    output_url = output_df['url'].tolist()[0]
    
    return output_url
    
def avg_feature_vector(sentence, model, num_features, index2word_set):
    
    """
    Function that calculates the average vector for 
    all words in a sentence
    
    Note: This function has been referred from 
    StackOverflow
    
    Parameters:
    sentence (str) : sentence for which average of word vectors
                     is to be calculated
    model (Word2Vec model): Word2Vec model trained on the related corpus
    num_features (int) : Maximum number of features to be considered for
                         averaging
    index2word_set (list) : list of words in the vocabulary
    
    Returns:
    feature_vec (float) : average over the vector for the sentence
    """    
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec
