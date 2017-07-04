import os
import time
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd

from dataset_merger.read_datasets import read_datasets
from dataset_merger.read_datasets import read_google_art_artists_info


warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


Dataset = namedtuple('Dataset', ['name', 'df'])
output_dir = '/export/home/asanakoy/tmp/dataset_merger_res_test'






def match_artists(output_dir, dfs):
    artists = dict()
    for name, df in dfs.iteritems():
        print name
        artists[name] = pd.DataFrame(
            data={'artist_name': sorted(df['artist_name'].unique()), 'wiki_url': np.nan})
        if name == 'googleart':
            # use scraped wiki_url s for googleart's artists
            tmp_artists_df = read_google_art_artists_info()
            tmp_artists_df = tmp_artists_df.dropna(subset=['wiki_url'])
            tmp_artists_df.index = tmp_artists_df['name']
            assert not tmp_artists_df.index.has_duplicates
            cnt = pd.notnull(tmp_artists_df.loc[artists[name]['artist_name'], 'wiki_url']).sum()
            assert cnt > 0, cnt
            artists[name].loc[:, 'wiki_url'] = \
                tmp_artists_df.loc[artists[name]['artist_name'], 'wiki_url'].values


def run(output_dir, dfs):
    print 'Merging:', dfs.keys()
    stack = [Dataset(*x) for x in dfs.iteitems()]
    num_el = len(stack)
    for i in xrange(num_el-1):
        if len(stack) == 1:
            break
        datasets = stack[-2:]
        features = [None, None]
        for k in xrange(2):
            print 'Compute features: {}'.format(datasets[k].name)
            # features[k] = feature_extractor(datasets[k].df, 'artist_50', output_dir, datasets[k].name)



if __name__ == '__main__':
    time_start = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dfs = read_datasets(['artuk', 'googleart', 'wga', 'wiki'])
    match_artists(output_dir, dfs)
    run(output_dir, dfs)
    print 'Elapsed time: {:.2f} sec'.format(time.time() - time_start)
