import os
from os.path import join
import pandas as pd
from collections import OrderedDict

import dataset_merger.read_datasets
from dataset_merger.prepare_artists import get_artists_with_years

datasets = ['wiki', 'googleart', 'wga', 'meisterwerke']
dfs = dataset_merger.read_datasets.read_datasets(datasets)

artists_with_years_dict = OrderedDict()
artists_with_years_dict['wiki'] = get_artists_with_years('wiki', dfs)
artists_with_years_dict['googleart'] = get_artists_with_years('googleart', dfs)
artists_with_years_dict['wga'] = get_artists_with_years('wga', dfs)
artists_with_years_dict['meisterwerke'] = get_artists_with_years('meisterwerke', dfs)


def change_artist_id(artist_id):
    dataset_name = None
    new_artist_id = None
    for name, df in artists_with_years_dict.iteritems():
        new_artist_id = u'{}_{}'.format(name, artist_id)
        if new_artist_id in df.index:
            dataset_name = name
            break
    assert new_artist_id is not None, artist_id
    return new_artist_id


if __name__ == '__main__':
    dir_path = '/export/home/asanakoy/workspace/dataset_merger/aggregated/'
    out_dir_path = '/export/home/asanakoy/workspace/dataset_merger/aggregated/renamed_manually_corrected_matches'
    dataset_names = ['wiki', 'googleart', 'wga', 'meisterwerke']

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    for i in xrange(1, len(dataset_names)):
        other_prefix = '+'.join(dataset_names[:i])
        print other_prefix, dataset_names[i]

        filepath = join(dir_path, '{}-{}_manually_corrected_matches.csv'.format(other_prefix, dataset_names[i]))
        df = pd.read_csv(filepath, encoding='utf-8', index_col=0)

        df['artist_id_{}'.format(dataset_names[i])] = df['artist_id_{}'.format(dataset_names[i])].apply(lambda x: u'{}_{}'.format(dataset_names[i], x))

        df['artist_id_{}'.format(other_prefix)] = df['artist_id_{}'.format(other_prefix)].apply(change_artist_id)

        df.to_csv(join(out_dir_path,
                       '{}-{}_manually_corrected_matches.csv'.format(other_prefix,
                                                                     dataset_names[i])), encoding='utf-8')