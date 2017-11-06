import os
from os.path import join, expanduser
import time
import warnings
from collections import namedtuple
from collections import defaultdict
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import deepdish as dd
import copy

import art_datasets.read
import artnet.info
import artnet.info.generate_artists_info
from dataset_merger import combine_objects
from dataset_merger.read_datasets import read_datasets
from eval.feature_extractor_tf import FeatureExtractorTf

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

ARTISTS_VERSION = '1.01'
VERSION = '1.01'
output_dir = expanduser('~/workspace/artnet/info')


class ImageHandle(object):
    def __init__(self, dataset_name, image_id, dist_in_group=None):
        """
        Args:
            dataset_name: name pf the dataset
            image_id: image id in teh current dataset
            dist_in_group: distance to the reference image in the current duplicate group (if any)

        """
        self.dataset_name = dataset_name
        self.image_id = image_id
        self.dist_in_group = dist_in_group

    def copy(self):
        return copy.copy(self)

    def __getitem__(self, item):
        return [self.dataset_name, self.image_id, self.dist_in_group][item]


def get_duplicate_groups(image_handles, dist_matrix, threshold_distance=0.066):
    """

    Args:
        image_handles: list, contains image handles ImageHandle
        dist_matrix: matrix len(image_handles) x len(image_handles) of distances between images
        threshold_distance: dist which to consider as duplicates. Distance lies in range [0, 2].
    Returns:
         image_duplicate_groups: list, each element is an image group.
            Image group is a list of image handles which are considered as duplicates.

    """
    image_duplicate_groups = list()
    for i in tqdm(xrange(len(image_handles)), desc='search for unique images'):
        cur_dataset = image_handles[i].dataset_name
        if dist_matrix[i, i] == 0:  # image was not assigned to any group yet
            cur_images_group = [image_handles[i].copy()]
            # To keep track the number of images per dataset in teh current group.
            # Don't remove add to the group more than one image from the same dataset.
            cur_group_images_per_dataset = defaultdict(int)

            for j in xrange(i + 1, len(image_handles)):
                duplicate_dataset = image_handles[j].dataset_name
                if (dist_matrix[i, j] <= threshold_distance and cur_dataset != duplicate_dataset and
                   cur_group_images_per_dataset[duplicate_dataset] == 0):
                    cur_group_images_per_dataset[duplicate_dataset] += 1
                    new_image_handle = ImageHandle(image_handles[j].dataset_name,
                                                   image_handles[j].image_id,
                                                   dist_in_group=dist_matrix[i, j])
                    cur_images_group.append(new_image_handle)
                    dist_matrix[j, :] = 100  # eliminate row
                    dist_matrix[:, j] = 100  # eliminate row
            image_duplicate_groups.append(cur_images_group)
    return image_duplicate_groups


def get_unique_images(image_handles, artist_id, snapshot_path=None, extractor=None):
    """

    Args:
        image_handles: list, contains image handles ImageHandle
        snapshot_path: net snapshot path to use for feature extraction

    Returns:
        image_handles: list of image handles ImageHandle representing unique images (without duplicates).
            Essentially contains the first element from each image duplicate group.
        image_duplicate_groups: list, each element is an image group.
            Image group is a list of image handles which are considered as duplicates.
    """

    num_datasets = len(np.unique([x[0] for x in image_handles]))
    if num_datasets == 1:
        return image_handles, [[x] for x in image_handles]

    image_paths = list()
    for handle in image_handles:
        module_info = importlib.import_module(handle.dataset_name + '.info')
        image_paths.append(join(module_info.crops_dir(),
                                handle.image_id + u'.{}'.format(art_datasets.read.image_ext(
                                    handle.dataset_name))))

    net_args = {'gpu_memory_fraction': 0.75,
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

    image_duplicate_groups = get_duplicate_groups(image_handles, dist_matrix)
    total_images_used = len([x for gr in image_duplicate_groups for x in gr])
    assert total_images_used == len(image_handles), '{}:: {} != {}'.format(
        artist_id, total_images_used, len(image_handles))

    return [x[0] for x in image_duplicate_groups], image_duplicate_groups


def group_duplicates(dfs, artists_df, snapshot_path, output_dir):
    """
    Group duplicate artworks
    Args:
        dfs:
        artists_df:
        snapshot_path:
        output_dir:

    Returns:
        dictionary, {artist_id: (list of groups of duplicate works)}

    """
    save_path = join(output_dir, 'unique_images_per_artist_v{}.h5'.format(VERSION))
    if os.path.exists(save_path):
        image_duplicate_groups = dd.io.load(save_path)
        return image_duplicate_groups

    net_args = {'gpu_memory_fraction': 0.96,
                'conv5': 'conv5/conv:0', 'fc6':
                    'fc6/fc:0', 'fc7': 'fc7/fc:0'}
    extractor = FeatureExtractorTf(snapshot_path, feature_norm_method=None,
                                   mean_path=None,
                                   net_args=net_args)

    # image_handles['some-artist-id-from-the-merged-dataset'] is a list of
    # pairs [(original_dataset_name, image_id), ...]. Essentially it's a list of all images
    # (counting duplicates as well) belonging to a specific artist.
    image_handles = dict()
    image_duplicate_groups = dict()
    print 'merged artists:', len(artists_df)
    for row in tqdm(artists_df[:].itertuples(), total=len(artists_df)):
        #     print row.artist_id, row.artist_ids
        image_handles[row.artist_id] = list()
        for artist_id in row.artist_ids:
            dataset_name, artist_id_short = artist_id.split('_', 1)
            cur_image_ids = dfs[dataset_name].index[
                dfs[dataset_name]['artist_id'] == artist_id_short]
            cur_image_handles = map(lambda x: ImageHandle(dataset_name=dataset_name, image_id=x),
                                    cur_image_ids)
            #         print '{}: {}'.format(artist_id_short, len(cur_image_ids))
            image_handles[row.artist_id].extend(cur_image_handles)
        _, image_duplicate_groups[row.artist_id] = get_unique_images(image_handles[row.artist_id],
                                                                   row.artist_id,
                                                                   snapshot_path, extractor=extractor)

    import tables
    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
    dd.io.save(save_path, image_duplicate_groups)
    print 'Saved.'
    return image_duplicate_groups


def get_artwork_object(dfs, image_handle):
    obj = dfs[image_handle.dataset_name].loc[[image_handle.image_id]].to_dict(orient='records')[0]
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


def create_artworks_df(dfs, artists_df, image_duplicate_groups_per_artist, output_dir):
    """
    Create a dataframe with unique works info.
    Merge metadata of duplicated works and remove duplicates.
    Args:
        dfs:
        artists_df:
        image_duplicate_groups_per_artist:
        output_dir:

    Returns:
        unique_artworks_df: DataFrame, containing info about unique works.

    """
    # key=lambda tup: tup[1])
    priority = {
        'wikiart': 0,
        'wiki': 0,
        'googleart': 1,
        'rijks': 1,
        'moma': 2,
        'artuk': 3,
        'wga': 4,
        'meisterwerke': 5,
    }
    new_artwork_objects = list()
    for artist_id, image_groups in tqdm(image_duplicate_groups_per_artist.iteritems()):
        for group in image_groups:
            # To take prototype for group not from 'wga' and 'meisterwerke' if possible we sort
            # handles according to dataset priority
            group.sort(key=lambda g: priority[g[0]])
            objects = [get_artwork_object(dfs, x) for x in group]
            new_object = combine_artworks(objects)
            new_object['artist_id'] = artist_id
            new_object['artist_name'] = artists_df.loc[artist_id, 'artist_name']
            new_object['years_range'] = artists_df.loc[artist_id, 'years_range']
            new_object['image_id'] = group[0].image_id
            new_object['source'] = group[0].dataset_name
            new_object['duplicate_ids'] = map(lambda x: x.dataset_name + '_' + x.image_id, group)
            new_object['duplicate_dists'] = map(lambda x: x.dist_in_group, group)

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

    dataset_name = 'wiki+googleart+wga+meisterwerke+moma+artuk+rijks'
    # artists_df can be generated with match_artists.ipynb
    artists_df = pd.read_hdf(
        join(output_dir, 'artists_{}_v{}.hdf5'.format(
            dataset_name, ARTISTS_VERSION)))

    datasets = ['wiki', 'artuk', 'googleart', 'moma', 'rijks', 'wga', 'meisterwerke']
    dfs = read_datasets(datasets)

    image_duplicate_groups = group_duplicates(dfs, artists_df, snapshot_path, output_dir)
    unique_artworks_df = create_artworks_df(dfs, artists_df, image_duplicate_groups, output_dir)
    assert not unique_artworks_df.index.has_duplicates
    unique_artworks_df.to_hdf(join(output_dir, 'artworks_{}_v{}.hdf5'.format(dataset_name,
                                                                             VERSION)),
                              'df', mode='w')
    print 'Elapsed time: {:.2f} sec'.format(time.time() - time_start)
    artnet.info.generate_artists_info.main(output_version='v' + VERSION)


if __name__ == '__main__':
    main()
