from sklearn.metrics.pairwise import cosine_similarity
import deepdish as dd
import numpy as np
import os

def compute_similarity(data_folder, first_fea, second_fea, base_name_first, base_name_second):
    """
    compute similarity
    Args:
        data_folder: folder for saving data
        first_fea: fist feature list
        second_fea:second feature list
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'

    Returns:similarity matrix

    """
    num_sim = first_fea.shape[0]
    predictions = np.zeros(num_sim)
    for i in range(first_fea.shape[0]):
        predictions[i] = cosine_similarity(first_fea[i].reshape(1,-1), second_fea[i].reshape(1,-1))
    # save predictions
    file_name = 'similarity_' + base_name_first + '_' + base_name_second + '.h5'
    if not os.path.exists(os.path.join(data_folder, file_name)):
        dd.io.save(os.path.join(data_folder, file_name), predictions)
        print "saved similarity!"
    return predictions