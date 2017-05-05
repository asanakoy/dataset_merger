from sklearn.metrics import pairwise_distances
import deepdish as dd
import numpy as np
import time
import os
from os.path import join


def compute_similarity(data_folder, first_fea, second_fea, base_name_first, base_name_second,
                       n_jobs=4):
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

    print 'Computing sim {} x {}'.format(len(first_fea), len(second_fea))
    file_path = join(data_folder, 'similarity_{}_{}.h5'.format(base_name_first, base_name_second))
    if os.path.exists(file_path):
        print 'Loading cached sim matrix'
        predictions = dd.io.load(file_path)
    else:
        start = time.time()
        predictions = 1 - pairwise_distances(first_fea, second_fea, metric='cosine', n_jobs=n_jobs)
        end = time.time()
        elapsed = end - start
        print "it took {:.2f} minutes to compute similarity.".format(elapsed / 60.)
        dd.io.save(file_path, predictions)
        print "saved similarity!"
    return predictions


if __name__ == '__main__':
    data_folder = '/export/home/jli/workspace/data_after_run/'
    first_fea = np.load(os.path.join(data_folder, 'features_wiki.npy'))
    second_fea = dd.io.load(os.path.join(data_folder, 'features_moma.h5'))
    base_name_first = 'wiki'
    base_name_second = 'moma'
    compute_similarity(data_folder, first_fea, second_fea, base_name_first, base_name_second)