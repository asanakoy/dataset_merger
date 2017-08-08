import os
from os.path import join
import time
import warnings
from collections import namedtuple
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import deepdish as dd

import art_datasets.read
from dataset_merger import combine_objects
from dataset_merger.read_datasets import read_datasets
from dataset_merger.prepare_artists import get_artists_with_years
from eval.feature_extractor_tf import FeatureExtractorTf

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


output_dir = '/export/home/asanakoy/tmp/dataset_merger_res_test'


def get_duplicate_groups(image_handles, dist_matrix):
    # todo: do not remove more than one duplicate from the same dataset

    unique_image_groups = list()
    for i in tqdm(xrange(len(image_handles)), desc='search for unique images'):
        if dist_matrix[i, i] == 0:
            cur_images_group = [image_handles[i]]
            for j in xrange(i + 1, len(image_handles)):
                if dist_matrix[i, j] <= 0.1 and image_handles[i][0] != image_handles[j][0]:  # 5% of closest, different datasets
                    cur_images_group.append(image_handles[j])
                    dist_matrix[j, :] = 100  # eliminate row
                    dist_matrix[:, j] = 100  # eliminate row
            unique_image_groups.append(cur_images_group)
    return unique_image_groups


def get_unique_images(image_handles, artist_id, snapshot_path=None, extractor=None):
    """

    Args:
        image_handles: lsit of image handles (dataset_name, image_id)
        snapshot_path: net snapshot path

    Returns:
        - list of unique image handles
        - list of unique image groups (each group contain all handles of duplicates)
    """

    num_datasets = len(np.unique([x[0] for x in image_handles]))
    if num_datasets == 1:
        return image_handles, [[x] for x in image_handles]

    # TODO: calculate between blocks only

    image_paths = list()
    for handle in image_handles:
        dataset_name, image_id = handle
        image_paths.append(join(art_datasets.read.crops_dir[dataset_name],
                                image_id + u'.{}'.format(art_datasets.read.image_ext(dataset_name))))

    net_args = {'gpu_memory_fraction': 0.96,
                'conv5': 'conv5/conv:0', 'fc6':
                    'fc6/fc:0', 'fc7': 'fc7/fc:0'}
    layers = ['fc7']
    if extractor is None:
        if snapshot_path is None:
            raise ValueError('snapshot_path must be not None')
        extractor = FeatureExtractorTf(snapshot_path, feature_norm_method=None,
                                       mean_path=None,
                                       net_args=net_args)

    # extract features from image_path
    features = extractor.extract(image_paths, layer_names=layers, verbose=0)
    dist_matrix = pairwise_distances(features, metric='cosine', n_jobs=4)
    np.fill_diagonal(dist_matrix, 0)
    assert np.max(dist_matrix) <= 2.0, np.max(dist_matrix)
    assert np.all(np.diagonal(dist_matrix) == 0), np.diagonal(dist_matrix)

    unique_image_groups = get_duplicate_groups(image_handles, dist_matrix)
    total_images_used = len([x for gr in unique_image_groups for x in gr])
    assert total_images_used == len(image_handles), '{}:: {} != {}'.format(
        artist_id, total_images_used, len(image_handles))

    return [x[0] for x in unique_image_groups], unique_image_groups


def group_duplicates(dfs, artists_df, snapshot_path, output_dir):
    save_path = join(output_dir, 'unique_images_per_artist.h5')
    if os.path.exists(save_path):
        unique_image_handles = dd.io.load(save_path)
        return unique_image_handles

    net_args = {'gpu_memory_fraction': 0.96,
                'conv5': 'conv5/conv:0', 'fc6':
                    'fc6/fc:0', 'fc7': 'fc7/fc:0'}
    extractor = FeatureExtractorTf(snapshot_path, feature_norm_method=None,
                                   mean_path=None,
                                   net_args=net_args)

    image_handles = dict()
    unique_image_handles = dict()
    print 'merged artists:', len(artists_df)
    for row in tqdm(artists_df[:].itertuples(), total=len(artists_df)):
        #     print row.artist_id, row.artist_ids
        image_handles[row.artist_id] = list()
        for artist_id in row.artist_ids:
            dataset_name, artist_id_short = artist_id.split('_', 1)
            cur_image_ids = dfs[dataset_name].index[
                dfs[dataset_name]['artist_id'] == artist_id_short]
            cur_image_ids = zip([dataset_name] * len(cur_image_ids), cur_image_ids)
            #         print '{}: {}'.format(artist_id_short, len(cur_image_ids))
            image_handles[row.artist_id].extend(cur_image_ids)
        _, unique_image_handles[row.artist_id] = get_unique_images(image_handles[row.artist_id],
                                                                   row.artist_id,
                                                                   snapshot_path, extractor=extractor)

    import tables
    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
    dd.io.save(save_path, unique_image_handles)
    print 'Saved.'
    return unique_image_handles


def get_artwork_object(dfs, image_handle):
    dataset_name, image_id = image_handle
    obj = dfs[dataset_name].loc[[image_id]].to_dict(orient='records')[0]
    return obj


def combine_artworks(objects_list):
    objects_list = map(lambda x: defaultdict(lambda: np.nan, x), objects_list)
    for obj in objects_list:
        pass

    new_object = {}
    new_keys = set()
    map(lambda x: new_keys.update(x), [x.keys() for x in objects_list])
    new_keys -= {'artist_slug', 'years_bio', 'artist_names', 'artist_name', 'artist_url_wiki',
                 'born-died', 'date',
                 'artist_details', 'artist_name_extra', 'comment'}
    for key in new_keys:
        if key in ['keywords', 'list_of_styles']:
            new_object[key] = combine_objects.take_union(objects_list, key)
        else:
            new_object[key] = combine_objects.take_first(objects_list, key)
    return new_object


def create_artworks_df(dfs, artists_df, unique_image_handles, output_dir):
    objects = list()

    cnt = 0
    new_artwork_objects = list()
    for artist_id, image_groups in tqdm(unique_image_handles.iteritems()):
        for group in image_groups:
            objects = [get_artwork_object(dfs, x) for x in group]
            new_object = combine_artworks(objects)
            new_object['artist_id'] = artist_id
            new_object['artist_name'] = artists_df.loc[artist_id, 'artist_name']
            new_object['years_range'] = artists_df.loc[artist_id, 'years_range']
            new_object['image_id'] = group[0][1]
            new_object['source'] = group[0][0]
            new_artwork_objects.append(new_object)
    unique_artworks_df = pd.DataFrame.from_dict(new_artwork_objects)
    unique_artworks_df.index = unique_artworks_df['image_id']
    return unique_artworks_df


def main():
    snapshot_path = '/export/home/asanakoy/workspace/tfprj/data/bvlc_alexnet.tf'
    # snapshot_path_second = '/export/home/asanakoy/workspace/wikiart/cnn/artist_50/rs_balance/model/snap_iter_335029.tf'

    time_start = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_name = 'wiki+googleart+wga+meisterwerke'
    artists_df = pd.read_hdf(
        '/export/home/asanakoy/workspace/dataset_merger/aggregated/artists_{}_v0.9.hdf5'.format(
            dataset_name))

    datasets = ['wiki', 'googleart', 'wga', 'meisterwerke']
    dfs = read_datasets(datasets)

    artists_with_years_dict = dict()
    artists_with_years_dict['googleart'] = get_artists_with_years('googleart', dfs)
    artists_with_years_dict['wiki'] = get_artists_with_years('wiki', dfs)
    artists_with_years_dict['wga'] = get_artists_with_years('wga', dfs)
    artists_with_years_dict['meisterwerke'] = get_artists_with_years('meisterwerke', dfs)

    unique_image_handles = group_duplicates(dfs, artists_df, snapshot_path, output_dir)
    unique_artworks_df = create_artworks_df(dfs, artists_df, unique_image_handles, output_dir)
    unique_artworks_df.to_hdf(join(output_dir, 'artworks_{}_v0.9.hdf5'.format(dataset_name)),
                              'df', mode='w')

    print 'Elapsed time: {:.2f} sec'.format(time.time() - time_start)


if __name__ == '__main__':
    main()
