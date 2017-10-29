from collections import defaultdict
import itertools
from fuzzywuzzy import fuzz
import numpy as np
import os
import sys
from tqdm import tqdm
import pandas as pd
import regex as re
import urllib
import multiprocessing
from sklearn.utils import gen_even_slices

import make_data.dataset
import wikiart.info.preprocess_info
from art_utils.pandas_tools import is_null_object
from art_utils.text_tools import parse_bool
from art_utils.joblib_wrapper import ParallelTqdm, delayed
from art_utils.text_tools import extract_all_years
import dataset_merger.read_datasets
import combine_objects

# constant to encode values from the second artist dataframe
# when find connected components in the graph
SECOND_ARTIST_ENCODE_MULTIPLIER = int(1e6)


def get_years_range_sim(range_a, range_b, is_bio=False, max_dist=85.0, max_delta=10,
                        allow_include=True,
                        allow_second_range_with_nan=False, bio_range=None):
    """

    Args:
        range_a:
        range_b:
        is_bio:

    Returns: similarity score from 0 to 1.0

    """
    if (np.isnan(range_a[0]) or np.isnan(range_b[0])) and \
            (np.isnan(range_a[1]) or np.isnan(range_b[1])):
        raise ValueError('Cannot compare 2 ranges with nans: {}, {}'.format(range_a, range_b))

    if allow_second_range_with_nan:
        range_a, range_b = list(range_a), list(range_b)
        if np.isnan(range_b[0]):
            assert not np.isnan(range_b[1]), 'one of the dates must be not none!'
            range_b[0] = range_a[0]
            if range_b[0] > range_b[1] + 3:
                return 0
            max_dist = 0
        elif np.isnan(range_b[1]):
            assert not np.isnan(range_b[0]), 'one of the dates must be not none!'
            range_b[1] = range_a[1]
            if range_b[1] < range_b[0] - 3:
                return 0
            max_dist = 0

    assert isinstance(range_a, list) and isinstance(range_b, list)
    ranges = [range_a, range_b]
    ranges.sort()
    for i in xrange(2):
        ranges[i] = np.asarray(ranges[i], dtype=float)
    assert np.all(np.isfinite(ranges)), ranges

    if bio_range is not None:
        max_dist = 0  # disable if we have bio
        if allow_include and not np.isnan(bio_range[0]) and (ranges[0][0] < bio_range[0] - 1):
            allow_include = False
        if allow_include and not np.isnan(bio_range[1]) and (bio_range[1] + 1 < max(ranges[0][1], ranges[1][1])):
            allow_include = False

    if allow_include and ((ranges[0][0] <= ranges[1][0] and ranges[1][1] <= ranges[0][1]) or
                          (ranges[1][0] <= ranges[0][0] and ranges[0][1] <= ranges[1][1])):
        # one is subset of another
        sim = 1.0
    else:
        dist = abs(ranges[1][1] - ranges[0][0])
        delta = max(abs(ranges[1][1] - ranges[0][1]), abs(ranges[1][0] - ranges[0][0]))
        assert dist >= 0, ranges
        assert delta >= 0, ranges
        # TODO: process if artist is still alive (death > 2017 ~2099)
        # TODO: if one of the ranges is bio, allow only delta < 10
        if dist <= max_dist or delta <= max_delta:
            sim = 1.0
        else:
            if delta <= max_delta + 10:
                sim = 1 - (min((delta - max_delta), 10) / 10.0) ** 2
            else:
                sim = 1 - np.sqrt(min((dist - max_dist), 35) / 35.0)
    return sim


def get_names_sim(names_a, names_b):
    """

    Args:
        names_a:
        names_b:

    Returns: similarity score from 0 to 100

    """
    scores = list()
    for a in names_a:
        for b in names_b:
            scores.append(fuzz.token_set_ratio(a, b))
    return max(scores)


def get_sim_urls_wiki(url_a, url_b):
    urls = [url_a, url_b]
    for i in xrange(len(urls)):
        url = urllib.unquote(urls[i].encode('utf8')).decode('utf-8')
        url = url.strip().lower()
        tokens = url.split('wikipedia.org')
        if len(tokens) != 2:
            raise ValueError(u'Not a wiki link:{}'.format(url))
        urls[i] = tokens[1]
    return float(urls[0] == urls[1]) * 101


def compute_sim_for_row(row_a, ns, num_cols):
    if hasattr(ns, 'df'):
        artists_df_b = ns.df
    else:
        artists_df_b = ns
    cur_sim = np.zeros(num_cols)
    names_a = row_a.artist_names
    years_range_a = row_a.years_range
    for j, row_b in enumerate(artists_df_b.itertuples()):
        if hasattr(row_a, 'wikidata_qid') and hasattr(row_b, 'wikidata_qid') and \
                not is_null_object(row_a.wikidata_qid) and not is_null_object(row_b.wikidata_qid):
            cur_sim[j] = (row_a.wikidata_qid.strip().lower() == row_b.wikidata_qid.strip().lower()) * 101
        elif hasattr(row_a, 'url_wiki') and hasattr(row_b, 'url_wiki') and \
                not is_null_object(row_a.url_wiki) and not is_null_object(row_b.url_wiki) and \
                'wikipedia.org' in row_a.url_wiki and \
                'wikipedia.org' in row_b.url_wiki:
            cur_sim[j] = get_sim_urls_wiki(row_a.url_wiki, row_b.url_wiki)
        elif hasattr(row_a, 'artist_id_degruyter') and hasattr(row_b, 'artist_id_degruyter') and \
                not is_null_object(row_a.url_wiki) and not is_null_object(row_b.url_wiki):
            cur_sim[j] = (row_a.artist_id_degruyter == row_b.artist_id_degruyter) * 101
        else:
            names_b = row_b.artist_names
            years_range_b = row_b.years_range
            years_sim = get_years_range_sim(years_range_a, years_range_b, max_dist=-1)
            if years_sim < 0.5:
                cur_sim[j] = years_sim * 0.9
            else:
                names_sim = get_names_sim(names_a, names_b)
                cur_sim[j] = years_sim * names_sim
    return cur_sim


def compute_sim_for_block(rows, ns, num_cols):
    sim_rows = np.zeros((len(rows), num_cols), dtype=np.float32)
    for i, row in enumerate(rows.iterrows()):
        sim_rows[i] = compute_sim_for_row(row[1], ns, num_cols)
    return sim_rows


def compute_sim_matrix(keys, artists_with_years_dict, n_jobs=1, num_blocks=None):
    assert len(keys) == 2, keys
    second_df = artists_with_years_dict[keys[1]].copy()
    num_cols = len(second_df)
    if 'url_wiki' not in second_df:
        second_df['url_wiki'] = np.nan
    second_df = second_df[['artist_names', 'years_range', 'url_wiki', 'artist_id_degruyter', 'wikidata_qid']]

    # create a manager to share dataframe across the processes
    mgr = multiprocessing.Manager()
    ns = mgr.Namespace()
    ns.df = second_df

    if n_jobs == 1:
        sim_matrix = np.zeros([len(artists_with_years_dict[key]) for key in keys])
        for i, row_a in tqdm(enumerate(artists_with_years_dict[keys[0]].iterrows()),
                             total=len(artists_with_years_dict[keys[0]])):
            row_a = row_a[1]
            sim_matrix[i, :] = compute_sim_for_row(row_a, ns, num_cols)
    else:
        num_rows = len(artists_with_years_dict[keys[0]])
        if num_blocks is None:
            print 'Num tasks: {}'.format(num_rows)
            sim_rows = ParallelTqdm(n_jobs=n_jobs, max_nbytes=512, verbose=1)(total=num_rows)\
                (delayed(compute_sim_for_row)(row_a[1], ns, num_cols)
                 for row_a in artists_with_years_dict[keys[0]].iterrows())
        else:
            first_df = artists_with_years_dict[keys[0]]
            print 'Num tasks: {}'.format(num_blocks)
            assert num_blocks >= n_jobs
            slices = gen_even_slices(len(first_df), num_blocks)
            sim_rows = ParallelTqdm(n_jobs=n_jobs, max_nbytes=512, verbose=8)(total=num_blocks) \
                (delayed(compute_sim_for_block)(first_df.iloc[s], ns, num_cols) for s in slices)

        sim_matrix = np.vstack(sim_rows)
    return sim_matrix


def fix_sim_matrix(dataset_names, dfs_to_merge, sim_matrix, manually_checked_df):
    sim_matrix = sim_matrix.copy()
    manually_checked_df['is_same'] = manually_checked_df['is_same'].apply(parse_bool).astype(int)
    for row in manually_checked_df.iterrows():
        # row = vars(row)
        row = row[1]
        ids = [row['artist_id_' + dataset_names[0]], row['artist_id_' + dataset_names[1]]]

        indices = [None] * 2
        for i in xrange(2):
            # print u'{}: :{}:'.format(dataset_names[i], ids[i])

            found = np.nonzero(dfs_to_merge[i]['artist_ids'].apply(lambda x: ids[i] in x))[0]
            if len(found):
                assert len(found), u'{}: :{}:'.format(dataset_names[i], ids[i])
                indices[i] = found[0]
            else:
                indices[i] = None
                # print u'Not found {}: :{}:'.format(dataset_names[i], ids[i])
        if indices[0] is not None and indices[1] is not None:
            new_sim_value = row['is_same'] * (101 + int(row['score'] >= 100))
            sim_matrix[indices[0], indices[1]] = new_sim_value
            if row['score'] == 107:
                assert new_sim_value > 100, '{}: {}'.format(new_sim_value, row)
    return sim_matrix


def get_num_top_matches(sims, min_sim=0):
    if sims[0] < min_sim:
        k = 0
    else:
        k = 1
        for val in sims[1:]:
            if abs(val - sims[0]) < 0.01 or val >= 100:
                k += 1
            else:
                break
    return k


def generate_matches_for_manual_check(dataset_names, dfs_to_merge, sim, min_sim=85, max_sim=100,
                                      min_k=1,
                                      output_path=None, discard_exaclty_matched_dates=False,
                                      min_total_works_count=0):
    # todo: remove with exact year matches
    if len(dfs_to_merge) != 2:
        raise ValueError('I can merge only 2 dataframes')
    objects_to_check = list()
    indices = sim.argsort(axis=1)[:, ::-1]
    for i in xrange(len(indices)):
        k = get_num_top_matches(sim[i, indices[i]], min_sim=min_sim)
        if k >= min_k:
            for j in indices[i][:k]:
                if min_sim <= sim[i, j] < max_sim:
                    obj = {
                        'score': sim[i, j],
                        'names_' + dataset_names[0]: dfs_to_merge[0].iloc[i].at['artist_names'],
                        'dates_' + dataset_names[0]: dfs_to_merge[0].iloc[i].at['years_range'],
                        'artist_id_' + dataset_names[0]: dfs_to_merge[0].index[i],
                        'names_' + dataset_names[1]: dfs_to_merge[1].iloc[j].at['artist_names'],
                        'dates_' + dataset_names[1]: dfs_to_merge[1].iloc[j].at['years_range'],
                        'artist_id_' + dataset_names[1]: dfs_to_merge[1].index[j],
                        'is_same': sim[i, j] >= 100,
                        'idx': (i, j),
                        'total_works_count': sum([dfs_to_merge[k].iloc[pos].at['works_count'] for k, pos in zip(range(2), (i, j))])
                    }
                    if discard_exaclty_matched_dates and \
                        obj['dates_' + dataset_names[0]] == obj['dates_' + dataset_names[1]]:
                        pass
                    elif obj['total_works_count'] < min_total_works_count:
                        pass
                    else:
                        objects_to_check.append(obj)
                    # print u'{} {} ({}) = {} ({})'.format(sim[i, j],
                    #                                      dfs_to_merge[0].iloc[i].at['artist_names'],
                    #                                      dfs_to_merge[0].iloc[i].at['years_range'],
                    #                                      dfs_to_merge[1].iloc[j].at['artist_names'],
                    #                                      dfs_to_merge[1].iloc[j].at['years_range'])
    df_for_sabine = pd.DataFrame.from_dict(objects_to_check)
    print 'count:', len(df_for_sabine)
    if len(df_for_sabine):
        df_for_sabine = df_for_sabine[['score', 'names_' + dataset_names[0],
                                       'dates_' + dataset_names[0],
                                       'names_' + dataset_names[1],
                                       'dates_' + dataset_names[1],
                                       'is_same',
                                       'artist_id_' + dataset_names[0],
                                       'artist_id_' + dataset_names[1],
                                       'idx',
                                       'total_works_count']]
        # df_for_sabine.sort_values(by='score', inplace=True)
    print len(df_for_sabine)
    df_for_sabine.head()
    if output_path is None:
        output_path = '/export/home/asanakoy/workspace/tmp/{}_check_matches.csv'.format(
            '-'.join(dataset_names))
    df_for_sabine.to_csv(output_path, encoding='utf-8')
    return df_for_sabine


def find_connected_components(sim, min_sim=100):
    """
    Find connected_components in the sim matrix
    """
    def find_ccomp_idx(x, connected_components):
        # find index on connected component
        res_idx = None
        for idx, ccomp in enumerate(connected_components):
            if x in ccomp:
                res_idx = idx
                break
        assert res_idx is not None
        return res_idx

    # encode col ids as (col_num + 1e6)
    connected_components = [set([i]) for i in xrange(sim.shape[0])] + \
                           [set([j + int(1e6)]) for j in xrange(sim.shape[1])]
    print 'connected_components at the beginning:', len(connected_components)
    for i in tqdm(xrange(sim.shape[0])):
        for j in xrange(sim.shape[1]):
            if sim[i, j] >= min_sim:
                idx_row = find_ccomp_idx(i, connected_components)
                idx_col = find_ccomp_idx(j + int(1e6), connected_components)
                connected_components[idx_row] = connected_components[idx_row].union(
                    connected_components[idx_col])
                del connected_components[idx_col]
    print 'connected_components:', len(connected_components)
    return connected_components


def get_big_connected_components(connected_components):
    big_comp_wiki_googleart = [x for x in connected_components if len(x) > 2]
    return big_comp_wiki_googleart


def combine_artists(objects_list):
    objects_list = map(lambda x: defaultdict(lambda: np.nan, x), objects_list)
    for obj in objects_list:
        if 'artist_ids' not in obj:
            obj['artist_ids'] = obj['artist_id']

    new_object = {}
    new_keys = set()
    map(lambda x: new_keys.update(x), [x.keys() for x in objects_list])
    new_keys -= {'artist_slug', 'total_items_count', 'artist_id'}
    for key in new_keys:
        if key in ['artist_name', 'url_wiki', 'artist_id_degruyter', 'wikidata_qid']:
            new_object[key] = combine_objects.take_first(objects_list, key)
        elif key in ['years']:
            new_object[key] = combine_objects.take_union(objects_list, key, take_group_works=False)
        elif key in ['artist_names', 'artist_ids', 'page_url']:
            new_object[key] = combine_objects.take_union(objects_list, key, take_group_works=True)
        elif key in ['years_work', 'years_bio', 'years_range']:
            new_object[key] = combine_objects.merge_years_range(objects_list, key,
                                                fallback_to_group_works=True)
        elif key == 'is_group_work':
            new_object[key] = np.all(combine_objects.take_union(objects_list, key, take_group_works=True))
        elif key == 'works_count':
            new_object[key] = np.sum([obj[key] for obj in objects_list])
        else:
            new_object[key] = combine_objects.take_first(objects_list, key)
    return new_object


def get_merged_artists_df(dataset_names, dfs_to_merge, sim_matrix, split_big_components=False, allow_big_components=False):
    """

    Args:
        dataset_names:
        dfs_to_merge:
        sim_matrix:
        split_big_components: split big components (size > 2) on a number of singe-element components

    Returns:

    """
    assert not (split_big_components and allow_big_components)
    if len(dfs_to_merge) != 2:
        raise ValueError('I can merge only 2 dataframes')
    assert len(dataset_names) == len(dfs_to_merge)
    assert sim_matrix.shape == (len(dfs_to_merge[0]), len(dfs_to_merge[1])), sim_matrix.shape

    def get_artist_object(artist_code, dfs_to_merge):
        dataset_idx = artist_code // SECOND_ARTIST_ENCODE_MULTIPLIER
        artist_idx = artist_code % SECOND_ARTIST_ENCODE_MULTIPLIER
        artist_obj = dfs_to_merge[dataset_idx].iloc[[artist_idx]].to_dict(orient='records')[0]
        return artist_obj

    manually_checked_matches_path = 'manually_corrected_matches/{}_manually_corrected_matches.csv'.format(
        '-'.join(dataset_names))
    if os.path.exists(manually_checked_matches_path):
        manually_checked_matches_df = pd.read_csv(manually_checked_matches_path, index_col=0,
                                                  encoding='utf-8')
        sim_matrix = fix_sim_matrix(dataset_names,
                                    dfs_to_merge,
                                    sim_matrix,
                                    manually_checked_matches_df)

    connected_components = find_connected_components(sim_matrix, min_sim=100)
    big_components = [x for x in connected_components if len(x) > 2]
    print 'big_components number:', len(big_components)
    if split_big_components:
        small_comp = [x for x in connected_components if len(x) <= 2]
        connected_components = small_comp + [{el} for x in big_components for el in x]
        big_components = [x for x in connected_components if len(x) > 2]
    if not allow_big_components and len(big_components) != 0:
        raise ValueError('big components are not allowed: ({})'.format(len(big_components)))

    merged_artist_objects = list()
    for component in tqdm(connected_components):
        artist_objects = map(lambda x: get_artist_object(x, dfs_to_merge), component)
        new_object = combine_artists(artist_objects)
        merged_artist_objects.append(new_object)

    assert len(connected_components) == len(merged_artist_objects)

    def artist_objects_to_dataframe(artist_objects):
        df = pd.DataFrame.from_dict(artist_objects)
        df['artist_id'] = df['artist_ids'].apply(lambda x: x[0])
        df.index = df['artist_id']
        return df

    return artist_objects_to_dataframe(merged_artist_objects)
