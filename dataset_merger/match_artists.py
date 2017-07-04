from collections import defaultdict
import itertools
from fuzzywuzzy import fuzz
import numpy as np
import os
import sys
from tqdm import tqdm
import pandas as pd
import regex as re

import make_data.dataset
import wikiart.info.preprocess_info
from art_utils.pandas_tools import is_null_object
from art_utils.text_tools import extract_all_years
import dataset_merger.read_datasets


def get_years_range_sim(range_a, range_b, is_bio=False):
    ranges = [range_a, range_b]
    ranges.sort()
    for i in xrange(2):
        ranges[i] = np.asarray(ranges[i], dtype=float)
    if ranges[0][0] <= ranges[1][0] and ranges[0][1] >= ranges[1][1]:
        # one is subset of another
        sim = 1.0
    else:
        usual_life_span = 85.0
        dist = float(ranges[1][1] - ranges[0][0])
        delta = float(ranges[1][1] - ranges[0][1])
        assert dist >= 0
        assert delta >= 0
        # TODO: process if artist is still aive (death > 2017 ~2099)
        # TODO: if one of the ranges is bio, allow only delta < 10
        if dist <= usual_life_span or delta <= 10:
            sim = 1.0
        else:
            if delta <= 20:
                sim = 1 - (min((delta - 10), 10) / 10.0) ** 2
            else:
                sim = 1 - np.sqrt(min((dist - usual_life_span), 35) / 35.0)
    return sim


def get_names_sim(names_a, names_b):
    scores = list()
    for a in names_a:
        for b in names_b:
            scores.append(fuzz.token_set_ratio(a, b))
    return max(scores)


def compute_sim_matrix(keys, artists_with_years_dict):
    assert len(keys) == 2, keys
    sim_matrix = np.zeros([len(artists_with_years_dict[key]) for key in keys])
    for i, row_a in tqdm(enumerate(artists_with_years_dict[keys[0]].itertuples()),
                         total=len(artists_with_years_dict[keys[0]])):
        names_a = row_a.names
        years_range_a = row_a.years_range

        for j, row_b in enumerate(artists_with_years_dict[keys[1]].itertuples()):
            if hasattr(row_a, 'url_wiki') and hasattr(row_b, 'url_wiki') and \
                    not is_null_object(row_a.url_wiki) and not not is_null_object(row_b.url_wiki):
                sim_matrix[i, j] = float(
                    row_a.url_wiki.lower().strip() == row_b.url_wiki.lower().strip())
            else:
                names_b = row_b.names
                years_range_b = row_b.years_range
                years_sim = get_years_range_sim(years_range_a, years_range_b)
                names_sim = get_names_sim(names_a, names_b)
                sim_matrix[i, j] = years_sim * names_sim
    return sim_matrix


def fix_sim_matrix(dataset_names, dfs_to_merge, sim_matrix, manually_checked_df):
    sim_matrix = sim_matrix.copy()
    manually_checked_df['is_same'] = manually_checked_df['is_same'].apply(parse_bool).astype(int)
    for row in manually_checked_df.itertuples():
        row = vars(row)
        ids = [row['artist_id_' + dataset_names[0]], row['artist_id_' + dataset_names[1]]]

        indices = [None] * 2
        for i in xrange(2):
            # print '{}: :{}:'.format(dataset_names[i], ids[i])
            indices[i] = np.nonzero(dfs_to_merge[i].index.values == ids[i])[0][0]
        sim_matrix[indices[0], indices[1]] = row['is_same'] * (100 + int(row['score'] == 100))
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


def parse_bool(value):
    if isinstance(value, basestring):
        value = value.lower()
    try:
        return bool(int(value))
    except:
        if value == 'true':
            return True
        elif value == 'false':
            return False
        else:
            raise ValueError('canot convert to bool: {}'.format(value))


def generate_matches_for_manual_check(dataset_names, dfs_to_merge, sim, min_sim=85, max_sim=100,
                                      min_k=1,
                                      output_path=None, discard_exaclty_matched_dates=False):
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
                        'names_' + dataset_names[0]: dfs_to_merge[0].iloc[i].at['names'],
                        'dates_' + dataset_names[0]: dfs_to_merge[0].iloc[i].at['years_range'],
                        'artist_id_' + dataset_names[0]: dfs_to_merge[0].index[i],
                        'names_' + dataset_names[1]: dfs_to_merge[1].iloc[j].at['names'],
                        'dates_' + dataset_names[1]: dfs_to_merge[1].iloc[j].at['years_range'],
                        'artist_id_' + dataset_names[1]: dfs_to_merge[1].index[j],
                        'is_same': sim[i, j] >= 100,
                        'idx': (i, j)
                    }
                    if discard_exaclty_matched_dates and \
                        obj['dates_' + dataset_names[0]] == obj['dates_' + dataset_names[1]]:
                        pass
                    else:
                        objects_to_check.append(obj)
                    # print u'{} {} ({}) = {} ({})'.format(sim[i, j],
                    #                                      dfs_to_merge[0].iloc[i].at['names'],
                    #                                      dfs_to_merge[0].iloc[i].at['years_range'],
                    #                                      dfs_to_merge[1].iloc[j].at['names'],
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
                                       'idx']]
        # df_for_sabine.sort_values(by='score', inplace=True)
    print len(df_for_sabine)
    df_for_sabine.head()
    if output_path is None:
        output_path = '/export/home/asanakoy/workspace/tmp/{}_check_matches.csv'.format(
            '-'.join(dataset_names))
    df_for_sabine.to_csv(output_path, encoding='utf-8')
    return df_for_sabine


def get_merged_artists_df(dataset_names, dfs_to_merge, sim, min_sim=85):
    if len(dfs_to_merge) != 2:
        raise ValueError('I can merge only 2 dataframes')
    assert len(dataset_names) == len(dfs_to_merge)
    assert sim.shape == (len(dfs_to_merge[0]), len(dfs_to_merge[1])), sim.shape
    # match is when sim = 100
    # TODO: combine artist_dataframes according to matches

    match_map = defaultdict(list)
    matches_df = generate_matches_for_manual_check(dataset_names, dfs_to_merge, sim, min_sim=100,
                                                  max_sim=110,
                                                  min_k=1,
                                                  output_path=None)
    assert len(matches_df['artist_id_' + dataset_names[0]].unique()) == len(matches_df)
    matched = [set(), set()]
    for matched_pair in zip(matches_df['artist_id_' + dataset_names[0]].values,
                    matches_df['artist_id_' + dataset_names[1]].values):
        match_map[matched_pair[0]].append(matched_pair[1])
        for i in xrange(2):
            matched[i].insert(matched_pair[i])

    new_columns = set(dfs_to_merge[0].columns).union(set(dfs_to_merge[1].columns))
    merged_df = dfs_to_merge[0].copy('artist_id_' + dataset_names[0])

    pass