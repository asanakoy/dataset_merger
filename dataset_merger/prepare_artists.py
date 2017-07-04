import numpy as np
import os
import sys
import itertools
from tqdm import tqdm
import pandas as pd

from art_utils.pandas_tools import is_null_object
import read_datasets


def merge_years_bio_years_work(artists_df):
    artists_df = artists_df.copy()
    assert 'years_work' in artists_df.columns
    assert 'years_bio' in artists_df.columns
    artists_df['years_range'] = artists_df['years_bio']

    mask_bio = artists_df['years_bio'].apply(lambda x: isinstance(x, list) and x[0] == x[1])
    mask_work = artists_df['years_work'].apply(lambda x: isinstance(x, list) and x[0] < x[1])
    mask = np.logical_and(mask_bio, mask_work)
    artists_df.loc[mask, 'years_range'] = artists_df.loc[mask, 'years_work']

    for idx in tqdm(artists_df.index):
        y_bio = artists_df.at[idx, 'years_bio']
        y_work = artists_df.at[idx, 'years_work']

        if is_null_object(y_bio):
            artists_df.at[idx, 'years_range'] = y_work
        elif is_null_object(y_work):
            pass
        elif y_bio[0] == y_bio[1] and y_work[0] < y_work[1]:
            artists_df.at[idx, 'years_range'] = y_work
    return artists_df['years_range']


def make_names_list(row):
    assert not is_null_object(row['artist_name'])
    res = [row['artist_name']]
    names = row['names']
    if not is_null_object(names) and len(names):
        assert isinstance(names, list), names.__class__
        names = [x.strip().lower().replace('\n', '').replace('\r', '') for x in names if not is_null_object(x)]
        res.extend(names)
    return np.unique(res).tolist()


def join_names_to_list(sa, sb):
    res = list()
    for a, b in itertools.izip(sa, sb):
        cur_names = list()
        assert not is_null_object(a) or not is_null_object(b), '{}, {}'.format(a, b)
        for val in [a, b]:
            if isinstance(val, list):
                cur_names.extend(val)
            else:
                cur_names.append(val)
        cur_names = [x.strip().lower() for x in cur_names if not is_null_object(x) and x]
        if not cur_names:
            cur_names = np.nan
        else:
            cur_names = np.unique(cur_names).tolist()
        res.append(cur_names)
    return res


def create_years_range(years):
    years = [x for x in years if not np.isnan(x)]
    if years:
        if max(years) - min(years) > 500:
            median = np.median(years)
            radius = min(median - min(years), max(years) - median)
            years = [x for x in years if abs(x - median) <= radius]
    if not years:
        years_range = np.nan
    else:
        years_range = [min(years), max(years)]
    return years_range


def get_google_art_artists_df(df, artist_id_field):
    df = df.copy()
    unique_ids, unique_indices = np.unique(df[artist_id_field], return_index=True)

    artists_info_from_works = df.iloc[unique_indices].copy()
    artists_info_from_works.index = artists_info_from_works[artist_id_field]
    artists_info_from_works.sort_index(inplace=True)
    artists_info_from_works['years'] = df.groupby(artist_id_field)['year'].apply(list).loc[
        artists_info_from_works.index]

    artists_df = pd.read_hdf(
        '/export/home/asanakoy/workspace/googleart/info/valid_artists.hdf5')
    artists_df.index = artists_df.artist_id
    artists_df.sort_index(inplace=True)
    assert set(artists_info_from_works.index).issubset(artists_df.index)
    artists_df = artists_df.loc[artists_info_from_works.index]
    assert np.array_equal(artists_info_from_works.index, artists_df.index)

    artists_df['artist_name'] = artists_df['name'].str.lower()
    del artists_df['name']
    artists_df['names'] = join_names_to_list(artists_df['artist_name'],
                                             artists_info_from_works['artist_name_extra'])
    artists_df['names'] = artists_df['names'].apply(remove_after_somebody_from_names)

    artists_df['url_wiki'] = artists_df['url_wiki'].str.lower()

    artists_df['years'] = artists_info_from_works['years']
    artists_df['years_work'] = artists_df['years'].apply(create_years_range)

    print 'with url_wiki:', pd.notnull(artists_df['url_wiki']).sum()

    artists_df['years_range'] = merge_years_bio_years_work(artists_df)

    print len(artists_df)
    artists_df.dropna(subset=['years_range'], inplace=True)
    print len(artists_df)

    assert not artists_df.index.has_duplicates
    return artists_df


def assign_group_works_to_the_first(list_of_names):
    """
    Transform 'jeremias falck after johann liss after bernardo strozzi' -> 'jeremias falck'
    Args:
        list_of_names: list of names to apply transformation

    Returns:

    """
    if is_null_object(list_of_names):
        return list_of_names
    else:
        new_list_of_names = list()
        for name in list_of_names:
            new_name = name.split('&')[0].strip(' ,')
            new_list_of_names.append(new_name)
        return new_list_of_names


def remove_after_somebody_from_names(list_of_names):
    if is_null_object(list_of_names):
        return list_of_names
    else:
        new_list_of_names = list()
        for name in list_of_names:
            tokens = name.split('after', 1)
            new_name = tokens[0].strip()
            if len(list_of_names) == 1:
                assert len(list_of_names) > 1 or new_name, 'empty name after removed `after`: {} [{}]'.format(name, list_of_names)
            if new_name:
                new_list_of_names.append(new_name)
        assert len(new_list_of_names), 'Empty list after removed `after`: {}'.fromat(list_of_names)
        return new_list_of_names


def get_artists_df(df, artist_id_field, split_group_naems_on_ampersand=False):
    unique_ids, unique_indices = np.unique(df[artist_id_field], return_index=True)

    years = df.groupby(artist_id_field)['year'].apply(list)

    artists_df = pd.DataFrame(index=unique_ids, data={'years': years.loc[unique_ids].values,
                                                      'artist_name': df['artist_name'].values[
                                                          unique_indices]})

    artists_df['years_work'] = artists_df['years'].apply(lambda x: [min(x), max(x)])
    artists_df['years_work'] = artists_df['years'].apply(create_years_range)

    if 'years_bio' in df.columns:
        artists_df['years_bio'] = df['years_bio'].values[unique_indices]
        artists_df['years_range'] = merge_years_bio_years_work(artists_df)
    else:
        artists_df['years_range'] = artists_df['years_work']
    if 'artist_names' in df.columns:
        artists_df['names'] = join_names_to_list(artists_df['artist_name'],
                                                 df['artist_names'].values[unique_indices])
    else:
        artists_df['names'] = join_names_to_list(artists_df['artist_name'],
                                                 artists_df['artist_name'])
    if split_group_naems_on_ampersand:
        artists_df['names'] = artists_df['names'].apply(assign_group_works_to_the_first)

    if 'artist_url_wiki' in df.columns:
        # .lower() because I downlaoded in lower for wikiart
        artists_df['url_wiki'] = df['artist_url_wiki'].iloc[unique_indices].str.lower()

    print len(artists_df)
    artists_df.dropna(subset=['years_range'], inplace=True)
    print len(artists_df)
    return artists_df


artist_id_field_map = {
    'wga': 'artist_id',
    'artuk': 'artist_id',
    'wiki': 'artist_slug',
    'googleart': 'artist_id'
}


get_artist_df_fn = {
    'wga': get_artists_df,
    'artuk': lambda a, b: get_artists_df(a, b, split_group_naems_on_ampersand=True),
    'wiki': get_artists_df,
    'googleart': get_google_art_artists_df
}


def get_artists_with_years(dataset_name, dfs):
    df = dfs[dataset_name]
    artist_id_field = artist_id_field_map.get(dataset_name, 'artist_name')
    artists_df = get_artist_df_fn[dataset_name](df, artist_id_field)
    return artists_df


def main():
    dfs = read_datasets.read_datasets()

    artists_with_years_dict = dict()
    artists_with_years_dict['wiki'] = get_artists_with_years('wiki', dfs)
    artists_with_years_dict['wga'] = get_artists_with_years('wga', dfs)
    artists_with_years_dict['artuk'] = get_artists_with_years('artuk', dfs)
    artists_with_years_dict['googleart'] = get_artists_with_years('googleart', dfs)


if __name__ == '__main__':
    main()
