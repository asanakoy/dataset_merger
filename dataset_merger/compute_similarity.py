from sklearn.metrics.pairwise import cosine_similarity
import deepdish as dd
import numpy as np
import time
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
    start = time.time()
    predictions = cosine_similarity(first_fea, second_fea)
    end = time.time()
    elapsed = end - start
    print "it takes " + `elapsed / 60` + " minutes to compute similarity."
    # save predictions
    file_name = 'similarity_' + base_name_first + '_' + base_name_second + '.h5'
    dd.io.save(os.path.join(data_folder, file_name), predictions)
    print "saved similarity!"
    return predictions


if __name__ == '__main__':
    data_folder = '/export/home/jli/workspace/data_after_run/'
    first_fea = np.load(os.path.join(data_folder, 'features_wiki.npy'))
    second_fea = dd.io.load(os.path.join(data_folder, 'features_moma.h5'))
    base_name_first = 'wiki'
    base_name_second = 'moma'
    compute_similarity(data_folder, first_fea, second_fea, base_name_first, base_name_second)