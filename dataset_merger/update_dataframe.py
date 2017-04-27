## -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import deepdish as dd
import os

def update_wiki(data_folder, wiki, moma, duplicates_list):
    """
    update wiki df info
    Args:
        data_folder: folder for saving data
        wiki: wiki df
        moma: moma df
        duplicates_list: duplicate list for moma

    Returns:updated version of wiki df
    """
    # add new columns for wiki
    wiki['dimensions'] = "" #np.nan
    wiki['credit_line'] = "" #np.nan
    wiki['moma_number'] ="" #np.nan
    wiki['classification'] = "" #np.nan
    wiki['department'] = "" #np.nan
    wiki['date_acquired'] = "" #np.nan
    wiki['curator_approved'] = "" #np.nan
    wiki['is_duplicate'] = False # means not duplicates
    wiki['dimensions'] = wiki['dimensions'].astype(object)
    wiki['credit_line'] = wiki['credit_line'].astype(object)
    wiki['moma_number'] = wiki['moma_number'].astype(object)
    wiki['classification'] = wiki['classification'].astype(object)
    wiki['department'] = wiki['department'].astype(object)
    wiki['date_acquired'] = wiki['date_acquired'].astype(object)
    wiki['curator_approved'] = wiki['curator_approved'].astype(object)
    # convert data format
    wiki['artist_slug'] = wiki['artist_slug'].astype(str)
    wiki['image_id'] = wiki.image_id.astype(object)
    wiki['dimensions'] = wiki.dimensions.astype(object)
    wiki['credit_line'] = wiki.credit_line.astype(object)
    wiki['moma_number'] = wiki.moma_number.astype(object)
    wiki['classification'] = wiki.classification.astype(object)
    wiki['department'] = wiki.department.astype(object)
    wiki['date_acquired'] = wiki.date_acquired.astype(object)
    wiki['curator_approved'] = wiki.curator_approved.astype(object)

    num_iters = len(moma)
    # append moma into wiki
    for i in range(num_iters):
        moma_image_id = moma.index[i] ## common
        # if moma is duplicate
        if str(moma_image_id) in duplicates_list[:,1]:
            # then update wiki info with moma
            loc = np.where(duplicates_list[:,1] == str(moma_image_id))
            wiki_id = duplicates_list[loc,0]
            wiki_id = str(wiki_id[0][0])
            moma_dimensions = moma[moma.index == moma_image_id].Dimensions.values[0]
            moma_credit_line = moma[moma.index == moma.index[i]].CreditLine.values[0]
            moma_moma_number = moma[moma.index == moma.index[i]].MoMANumber.values[0]
            moma_classification = moma[moma.index == moma.index[i]].Classification.values[0]
            moma_department = moma[moma.index == moma.index[i]].Department.values[0]
            moma_date_acquired = moma[moma.index == moma.index[i]].DateAcquired.values[0]
            moma_curator_approved = moma[moma.index == moma.index[i]].CuratorApproved.values[0]
            wiki.set_value(wiki_id, 'dimensions', moma_dimensions)
            wiki.set_value(wiki_id, 'credit_line', moma_credit_line)
            wiki.set_value(wiki_id, 'moma_number', moma_moma_number)
            wiki.set_value(wiki_id, 'classification', moma_classification)
            wiki.set_value(wiki_id, 'department', moma_department)
            wiki.set_value(wiki_id, 'date_acquired', moma_date_acquired)
            wiki.set_value(wiki_id, 'curator_approved', moma_curator_approved)
            wiki.set_value(wiki_id, 'is_duplicate', True)

    if not os.path.exists(os.path.join(data_folder, 'wiki_info_update.hdf5')):
        wiki.to_hdf(os.path.join(data_folder, 'wiki_info_update.hdf5'), 'image_id')
    print "wiki has updated!"
    return wiki


def update_w_m_based_rijks_dup(data_folder, rijks, wikimoma_merged, duplicates):
    """
    update wiki_moma based on duplicate part of rijks
    Args:
        data_folder:folder for saving data
        rijks:rijks df
        wikimoma_merged: merged wikimoma df
        duplicates:duplicate part for rijks

    Returns:updated version of wiki_moma

    """
    #print wikimoma_merged.columns
    #print duplicates.shape
    i = 154
    coverage = rijks[rijks.image_id == duplicates[i][1]].coverage.values[0]
    #print duplicates[i][0]
    d = int(duplicates[i][0])
    #print type(duplicates[i][0])
    wikimoma_merged['title'] = ""
    wikimoma_merged.set_value(d, 'coverage', coverage)
    #print duplicates[i][0].isdigit()
    for i in range(len(duplicates)):
        if duplicates[i][0].isdigit():
            coverage = rijks[rijks.image_id == duplicates[i][1]].coverage.values[0]
            wikimoma_merged.set_value(int(duplicates[i][0]), 'coverage', coverage)
            genre = rijks[rijks.image_id == duplicates[i][1]].genre.values[0]
            wikimoma_merged.set_value(int(duplicates[i][0]), 'genre', genre)
            size = rijks[rijks.image_id == duplicates[i][1]].size
            wikimoma_merged.set_value(int(duplicates[i][0]), 'size', size)
            title = rijks[rijks.image_id == duplicates[i][1]].title.values[0]
            wikimoma_merged.set_value(int(duplicates[i][0]), 'title', title)
            wikimoma_merged.set_value(int(duplicates[i][0]), 'is_duplicate', True)
        else:
            coverage = rijks[rijks.image_id == duplicates[i][1]].coverage.values[0]
            wikimoma_merged.set_value(duplicates[i][0], 'coverage', coverage)
            genre = rijks[rijks.image_id == duplicates[i][1]].genre.values[0]
            wikimoma_merged.set_value(duplicates[i][0], 'genre', genre)
            size = rijks[rijks.image_id == duplicates[i][1]].size
            wikimoma_merged.set_value(duplicates[i][0], 'size', size)
            title = rijks[rijks.image_id == duplicates[i][1]].title.values[0]
            wikimoma_merged.set_value(duplicates[i][0], 'title', title)
            wikimoma_merged.set_value(duplicates[i][0], 'is_duplicate', True)
    wikimoma_merged.to_hdf(os.path.join(data_folder, 'info_wikimoma_after_update_infomation.hdf5'), 'image_id')
    return wikimoma_merged


def update_same_artist_names(data_folder, wikipedia_list, subname_list, rijks):
    """
    update artist names for rijks
    Args:
        data_folder:folder for saving data
        wikipedia_list:same artist name list after checking at wikipedia
        subname_list:subname list for rijks
        rijks:rijks df

    Returns:updated version of rijks

    """
    for i in range(len(wikipedia_list)):
        rijks['artist_slug'] = rijks['artist_slug'].replace([wikipedia_list[i][1]], str(wikipedia_list[i][0]))
    for i in range(len(subname_list)):
        rijks['artist_slug'] = rijks['artist_slug'].replace([subname_list[i][1]], str(subname_list[i][0]))
    rijks.to_hdf(os.path.join(data_folder, 'rijks_info_after_unify_artist_names.hdf5'), 'image_id')
    return rijks


def merge_wikimoma_rijks(data_folder, wikimoma, rijks_unique):
    joined = pd.concat([wikimoma, rijks_unique], ignore_index=False)
    print joined.columns
    print len(rijks_unique)
    print len(wikimoma)
    print len(joined)
    joined.to_hdf(os.path.join(data_folder, 'info_wikimoma_rijks.hdf5'), 'image_id', format='fixed')
    return joined


def filter_genre(data_folder, rijks_unique):
    """
    filter genre
    Args:
        data_folder: folder for saving data
        rijks_unique: rijks unique part

    Returns:filter version of rijks based on keeping list
    Note: I can run this function successfully in my laptop with Python 2.7.13. But I fail to run it in this linux environment.
    I've coped result of this function from my laptop to this server. I think the different version of python lead to this
    bug.
    """
    list = ['prent', 'tekening', 'boekillustratie', 'ornamentprent', 'schilderij', 'historieprent', 'nieuwsprent', 'foto', 'ontwerp',\
        'kaart', 'titelpagina', 'spotprent', 'titelprent', 'surimono', 'propagandaprent', 'kostuumprent', 'stereofoto',\
        'albumblad', 'historisch objectvoorstelling', 'volksprent', 'schetsboek', 'Indiase miniatuur', 'embleem',\
        'embleem', 'grisaille', 'shunga', 'wandtapijt', 'ruit', 'tegel', 'portret', 'topografische tekening',\
        'titelblad', 'schildering', 'silhouet', 'karikatuur', 'pamflet', 'plaque', 'pastel', 'triptiek', 'dierstudie',\
        'schetsboekblad', 'nieuwjaarswens', 'kopie naar prent ', 'prentenalbum', 'album', 'zijpaneel', 'architectuurtekening',\
        'kostuumstudie', 'krant', 'panorama', 'rolschildering', 'familiewapen', 'kopie naar tekening ', 'achterglasschildering',\
        'sprei', 'vignet', 'kamerbeschildering', 'diptiek', 'magistratenkussen', 'Perzische miniatuur ', 'almanak', 'penschildering',\
        'papiernegatief', 'aquarel', 'ornamenttekening', 'huwelijksprent', 'muurschildering', 'sits', 'gouache', 'bidprent',\
        'decoratiestuk', 'schilderijlijst', 'verdure', 'documentaire foto', 'prenttekening', 'vidimus', 'vingerschildering',\
        'kamerscherm', 'beeldmotet', 'promotieprent', 'prentbriefkaart', 'cartografie', 'loterijprent']
    rijks_unique[rijks_unique['genre'].str.contains('prent', na=False)]
    rijks_unique = rijks_unique[rijks_unique['genre'].str.contains('|'.join(list), na=False)]
    rijks_unique.to_hdf(os.path.join(data_folder, 'rijks_unique_filter_genre.hdf5'), 'image_id')
    return rijks_unique


def add_source_column(data_folder, wiki, moma_unique,info):
    num_wiki = len(wiki)
    num_moma = len(moma_unique)
    num_rijks = len(info) - num_wiki - num_moma
    wiki_list = ['wiki'] * num_wiki
    moma_list = ['moma'] * num_moma
    rijks_list = ['rijks'] * num_rijks
    print len(info)
    print num_wiki
    print num_moma
    print num_rijks
    source_list = wiki_list + moma_list + rijks_list
    info['source'] = source_list
    info.to_hdf(os.path.join(data_folder, 'info_wiki_moma_rijks_final_result.hdf5'), 'image_id')


if __name__ == '__main__':
    # test
    data_folder = '/export/home/jli/workspace/readable_code_data/'
    #rijks_unique_filter = dd.io.load(os.path.join(data_folder, 'rijks_unique_filter_genre.hdf5'))
    #wikimoma = pd.read_hdf(os.path.join(data_folder, 'info_wikimoma_after_update_infomation.hdf5'))
    #merge_wikimoma_rijks(data_folder, wikimoma, rijks_unique_filter)
    wiki = pd.read_hdf(os.path.join(data_folder, 'wiki_info_update.hdf5'))
    moma_unique = pd.read_csv(os.path.join(data_folder, 'moma_info_unique.csv'), index_col='id')
    info = pd.read_hdf(os.path.join(data_folder, 'info_wikimoma_rijks.hdf5'))
    add_source_column(data_folder, wiki, moma_unique, info)
    ##wikimoma_merged = pd.read_hdf(os.path.join(data_folder, 'info_wiki_moma_merged.hdf5'))
    ##rijks_duplicates = dd.io.load('/export/home/jli/workspace/readable_code_data/split_info_wikimoma_rijks/duplicates_wikimoma_rijks_after_substr_detect.h5')
    ##rijks = pd.read_hdf(os.path.join(data_folder, 'rijks_info_after_unify_artist_names.hdf5'))
    ##update_w_m_based_rijks_dup(data_folder, rijks, wikimoma_merged, rijks_duplicates)
    '''
    #filter_genre(data_folder, rijks_unique)
    #rijks = pd.read_hdf(os.path.join(data_folder, 'rijks_info_after_unify_artist_names.hdf5'))
    #wikimoma_merged = pd.read_hdf(os.path.join(data_folder, 'info_wiki_moma_merged.hdf5'))
    #rijks_duplicates = dd.io.load('/export/home/jli/workspace/readable_code_data/split_info_wikimoma_rijks/duplicates_wikimoma_rijks_after_substr_detect.h5')
    #update_w_m_based_rijks_dup(data_folder, rijks, wikimoma_merged, rijks_duplicates)
    wikimoma = pd.read_hdf(os.path.join(data_folder, 'info_wikimoma_after_update_infomation.hdf5'))
    merge_wikimoma_rijks(data_folder, wikimoma, rijks_unique_filter)

    #wiki = pd.read_hdf(os.path.join(data_folder, 'wiki_info_update.hdf5'))
    #wikipedia_list = dd.io.load(os.path.join(data_folder, 'wikipedia_same_artists_wikimoma_rijks.h5'))
    #subname_list = dd.io.load(os.path.join(data_folder, 'sub_artist_names_w_r.h5'))
    #rijks = pd.read_hdf(os.path.join(data_folder, 'rijks_info.hdf5'))
    #rijks = update_same_artist_names(data_folder, wikipedia_list, subname_list, rijks)
    rijks_unique = dd.io.load(os.path.join(data_folder, 'rijks_unique_info.hdf5'))
    rijks_unique = filter_genre(data_folder, rijks_unique)
    #rijks_unique = pd.read_hdf(os.path.join(data_folder, 'rijks_unique_filter.h5'))
    #rijks = pd.read_hdf(os.path.join(data_folder, 'rijks_info_after_unify_artist_names.hdf5'))
    #print len(rijks)
    #wikimoma_merged = pd.read_hdf(os.path.join(data_folder, 'info_wiki_moma_merged.hdf5'))
    #rijks_duplicates = dd.io.load('/export/home/jli/workspace/readable_code_data/split_info_wikimoma_rijks/duplicates_wikimoma_rijks_after_substr_detect.h5')
    #update_w_m_based_rijks_dup(data_folder, rijks, wikimoma_merged, rijks_duplicates)
    #wikimoma = pd.read_hdf(os.path.join(data_folder, 'info_wikimoma_after_update_infomation.hdf5'))
    #rijks_unique_filter = dd.io.load(os.path.join(data_folder, 'rijks_unique_filter.h5'))
    #merge_wikimoma_rijks(data_folder, wikimoma, rijks_unique_filter)
    #moma_unique = pd.read_csv(os.path.join(data_folder, 'moma_info_unique.csv'), index_col='id')
    #info = pd.read_hdf(os.path.join(data_folder, 'info_wikimoma_after_update_infomation.hdf5'))
    #add_source_column(data_folder, wiki, moma_unique, rijks_unique, info)'''