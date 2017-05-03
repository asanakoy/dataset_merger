from Levenshtein import distance
import deepdish as dd
import pandas as pd
import numpy as np
import wikipedia
import requests
import time
import sys
import os
import re

def get_artist_names(data_folder, base_name_first, df_first, base_name_second, df_second):
    """
    get artist names for dataframes
    Args:
        data_folder: folder for saving data
        base_name_first: e.g. 'wiki'
        df_first: the first dataframe
        base_name_second: e.g. 'moma'
        df_second: the second dataframe

    Returns: artist name list from the first dataframe and the name list from the second dataframe

    """
    artists_first = np.unique(df_first.artist_slug.values).astype('U')
    artists_second = np.unique(df_second.Artist.values).astype('U')
    file_name_first = 'artists_' + base_name_first + '.h5'
    file_name_second = 'artists_' + base_name_second + '.h5'
    if not (os.path.exists(os.path.join(data_folder, file_name_first)) and os.path.exists(os.path.join(data_folder, file_name_second))):
        dd.io.save(os.path.join(data_folder, file_name_first), artists_first)
        dd.io.save(os.path.join(data_folder, file_name_second), artists_second)
        print 'artists has saved!'
    return artists_first, artists_second


def get_artist_names_sec(data_folder, base_name_first, df_first, base_name_second, df_second):
    """
    get artist name list for the second mergeing
    Args:
        data_folder: folder for saving data
        base_name_first: e.g. 'wiki'
        df_first: the first dataframe
        base_name_second: e.g. 'moma'
        df_second: the second dataframe

    Returns: artist name list from the first df and from the second df

    """
    artists_first = np.unique(df_first.artist_slug.values).astype('U')
    artists_second = np.unique(df_second.artist_slug.values).astype('U')
    file_name_first = 'artists_' + base_name_first + '.h5'
    file_name_second = 'artists_' + base_name_second + '.h5'
    if not (os.path.exists(os.path.join(data_folder, file_name_first)) and os.path.exists(os.path.join(data_folder, file_name_second))):
        dd.io.save(os.path.join(data_folder, file_name_first), artists_first)
        dd.io.save(os.path.join(data_folder, file_name_second), artists_second)
        print 'artists has saved!'
    return artists_first, artists_second


def check_substr(data_folder, artists_first, artists_second, base_name_first, base_name_second):
    """
    check if artist names from the first dataframe is the substr of artists form the second dataframe
    Args:
        data_folder:folder for saving data
        artists_first:artist names from the first df
        artists_second:artist names from the second df
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'

    Returns: substr artist name list

    """
    substr_list_number = []
    substr_artists_list = []

    for j in range(len(artists_first)):
        list = [i for i,item in enumerate(artists_second) if artists_first[j] in item]
        if list:
            for i in range(len(list)):
                substr_list_number.append([j, list[i]])
    # get artist name based on substr_list_number
    for i in range(len(substr_list_number)):
        wiki_ele = artists_first[substr_list_number[i][0]]
        moma_ele = artists_second[substr_list_number[i][1]]
        if wiki_ele != moma_ele:
            list_ele = [wiki_ele, moma_ele]
            substr_artists_list.append(list_ele)
    # delete false artist map
    substr_artists_list = del_false_artist_map(substr_artists_list)
    # save this list
    file_name = 'substr_artists_list_' + base_name_first + '_' + base_name_second + '.h5'
    if not os.path.exists(os.path.join(data_folder, file_name)):
        dd.io.save(os.path.join(data_folder, file_name), substr_artists_list)
        print 'substr artist list has saved!'
    return substr_artists_list


def update_artist_names(data_folder, update_artists_list, df_first, df_second, base_name_second):
    """
    update moma's duplicate artist names based on wiki's artist names
    Args:
        data_folder:folder for saving data
        update_artists_list:same artist name list
        df_first:the first dataframe
        df_second:the second dataframe
        base_name_second:e.g. 'moma'

    Returns: update this artist name list

    """
    for i in range(len(update_artists_list)):
        modify_artist = df_first[df_first['artist_slug'] == update_artists_list[i][0]].artist_slug.values[0]
        df_second['Artist'] = df_second['Artist'].replace([update_artists_list[i][1]], str(modify_artist))
    # update and save
    file_name = base_name_second + '_update_artist_names.csv'
    df_second.to_csv(os.path.join(data_folder, file_name))
    print 'update artist names'
    return df_second


def compute_text_distance(data_folder, artists_first, artists_second, base_name_first, base_name_second):
    """
    compute text distances between the first artist name list and the second artist name list
    Args:
        data_folder:folder for saving data
        artists_first: artist name list form the first df
        artists_second:artist name list form the second df
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'

    Returns: text distance matrix

    """
    row_distance = []
    num_iters = len(artists_first)
    for i in range(num_iters):
        col_distance = []
        for j in range(len(artists_second)):
            dis_wiki_moma = distance(artists_first[i], artists_second[j])
            col_distance.append(dis_wiki_moma)
        row_distance.append(col_distance)
    row_distance = np.array(row_distance)
    # save text distance matrix
    save_name = 'text_distances_' + base_name_first + '_' + base_name_second + '.h5'
    if not os.path.exists(os.path.join(data_folder, save_name)):
        dd.io.save(os.path.join(data_folder, save_name), row_distance)
        print "text distances has saved!"
    return row_distance


def compute_max_length_matirx(data_folder, artists_first, artists_second, base_name_first, base_name_second):
    """
    compute max length between the first artist name list and the second artist name list
    Args:
        data_folder:folder for saving data
        artists_first: artist name list form the first df
        artists_second:artist name list form the second df
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'

    Returns: maximum length text distance matrix

    """
    row_len = []
    num_iters = len(artists_first)
    for i in range(num_iters):
        col_len = []
        for j in range(len(artists_second)):
            if len(artists_first[i]) >= len(artists_second[j]):
                max_len = len(artists_first[i])
            else:
                max_len = len(artists_second[j])
            col_len.append(max_len)
        row_len.append(col_len)
    row_len = np.array(row_len)
    # save maximum length matrix for wiki and moma
    file_name = 'max_len_matrix_' + base_name_first + '_' + base_name_second + '.h5'
    dd.io.save(os.path.join(data_folder, file_name), row_len)
    return row_len

def compute_text_dis_rate(data_folder, max_len_matrix, text_dis_matrix, base_name_first, base_name_second):
    """
    normalize the text distance, 0 means from the same artist, 1 means from totally different artist
    Args:
        data_folder:folder for saving data
        max_len_matrix: max length text distance matrix
        text_dis_matrix:text distance matrix
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'

    Returns: normalization rate for text distance

    """
    row_rate = []
    num_iters = max_len_matrix.shape[0]
    for i in range(num_iters):
        col_rate = []
        for j in range(max_len_matrix.shape[1]):
            col_rate.append(text_dis_matrix[i][j] / (max_len_matrix[i][j] * 1.0))
        row_rate.append(col_rate)
    row_rate = np.array(row_rate)
    print row_rate.shape
    file_name = 'text_dis_rate_' + base_name_first + '_' + base_name_second + '.h5'
    if not os.path.exists(os.path.join(data_folder, file_name)):
        dd.io.save(os.path.join(data_folder, file_name), row_rate)
    return row_rate


def search_in_wikipedia(data_folder, rate_matrix, artists_first, artists_second, base_name_first, base_name_second):
    """
    search pairs of artist names in wikipedia and check if they are from the same artist
    Args:
        data_folder:folder for saving data
        rate_matrix:normalization rate of text distance
        artists_first: artist name list form the first df
        artists_second:artist name list form the second df
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'

    Returns: same artist name list after checking in wikipedia

    """
    threshold = 0.50
    match_artists = []
    num_iters = rate_matrix.shape[0]
    print rate_matrix.shape[0]
    print rate_matrix.shape[1]
    for i in range(num_iters):
        for j in range(rate_matrix.shape[1]):
            if rate_matrix[i][j] <= threshold and rate_matrix[i][j] != 0:
                match_artists.append([artists_first[i], artists_second[j]])
        sys.stdout.write("\r%d/%d" % (i, num_iters-1))
        sys.stdout.flush()

    num_data = len(match_artists)
    wikipedia_same_artists_list = []
    count = 0
    start = time.time()
    for item in match_artists:
        count += 1
        sys.stdout.write("\r%d/%d" % (count, num_data))
        sys.stdout.flush()
        try:
            #try to load the wikipedia page
            item_first = wikipedia.page(item[0])
            item_sec = wikipedia.page(item[1])
            if item_first == item_sec:
                print item
                print "True"
                wikipedia_same_artists_list.append(item)
        except wikipedia.exceptions.PageError:
            #if a "PageError" was raised, ignore it and continue to next link
            continue
        except wikipedia.exceptions.DisambiguationError:
            continue
        except requests.exceptions.ConnectionError:
            print "sleep once!"
            time.sleep(3)
    # save this list
    file_name = 'wikipedia_same_artists_' + base_name_first + '_' +base_name_second + '.h5'
    if not os.path.exists(os.path.join(data_folder, file_name)):
        dd.io.save(os.path.join(data_folder, file_name), wikipedia_same_artists_list)
    end = time.time()
    elapsed_whole = end - start
    print "it takes " + `elapsed_whole / 60` + " minutes to search same artists in wikipedia."
    return wikipedia_same_artists_list


def check_wikipedia(data_folder, artists_first, artists_second, base_name_first, base_name_second):
    """
    whole process for checking pairs of artist names in wikipedia
    Args:
        data_folder:folder for saving data
        artists_first: artist name list form the first df
        artists_second:artist name list form the second df
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'

    Returns: same artist name list after checking in wikipedia

    """
    text_dis_matrix = compute_text_distance(data_folder, artists_first, artists_second, base_name_first, base_name_second)
    max_len_matrix = compute_max_length_matirx(data_folder, artists_first, artists_second, base_name_first, base_name_second)
    rate_matrix = compute_text_dis_rate(data_folder, max_len_matrix, text_dis_matrix, base_name_first, base_name_second)
    wikipedia_same_artists_list = search_in_wikipedia(data_folder, rate_matrix, artists_first, artists_second, base_name_first, base_name_second)
    return wikipedia_same_artists_list

def del_false_artist_map(artists_list):
    """
    remove false artist map, e.g. ['ben', 'benjas-muller']. Actually they are from different artist names
    Args:
        artists_list: artist name list for detecting

    Returns: same artist name list after moving false pairs of artist names

    """
    del_indices = []
    for i in range(len(artists_list)):
        if len(re.split('[-,.()]', artists_list[i][0])) == 1:
                for part in re.split('[-,.()]', artists_list[i][1]):
                    if part != artists_list[i][0]:
                        # remove item from the list
                        del_indices.append(i)

    for i in sorted(list(set(del_indices)), reverse=True):
        del artists_list[i]
    return artists_list


def modify_false_neg_artist_list(data_folder, false_neg_artist_list):
    """
    check if they are substr for each other
    Args:
        false_neg_artist_list: false negative artist name list

    Returns: true same artist name list

    """
    same_artists = []
    unique_same_artist_list = []
    for i in range(len(false_neg_artist_list)):
        flag = False
        for item_fir in re.split('[-,.()]', false_neg_artist_list[i][0]):
            for item_sec in re.split('[-,.()]', false_neg_artist_list[i][1]):
                if flag is False and item_fir == item_sec:
                    same_artists.append(false_neg_artist_list[i].tolist())
                    flag = True
    for item in same_artists:
        if item not in unique_same_artist_list:
            unique_same_artist_list.append(item)
    dd.io.save(os.path.join(data_folder, 'unique_same_artist_list_wiki_moma.h5'), unique_same_artist_list)
    print 'unique_same_artist_list has saved!'
    return unique_same_artist_list


def detect_sub_artist_names(data_folder, false_pos_artists):
    """
    second method for checking if they are substr for each other
    Args:
        false_pos_artists:false positive artist name list

    Returns: same artist name list

    """
    # detect substr in rijks
    list = []
    done_list = []
    for i in range(len(false_pos_artists)):
        wikimoma_ori_artist = false_pos_artists[i][0]
        wikimoma_artist = false_pos_artists[i][0].lower()
        rijks_ori_artist = false_pos_artists[i][1]
        rijks_artist = false_pos_artists[i][1].lower()
        for char in wikimoma_artist:
            if not char.isalpha():
                #print char
                substr = wikimoma_artist.split(char,1)[0]
                break
            else:
                substr = wikimoma_artist.split(char,1)[0]
        if rijks_artist.find(substr) != -1:
            list.append([wikimoma_ori_artist, rijks_ori_artist])
            done_list.append(i)
    num_iters = len(false_pos_artists) - len(done_list)
    # detect substr in wikimoma
    for i in range(len(false_pos_artists)):
        if i not in done_list:
            wikimoma_ori_artist = false_pos_artists[i][0]
            wikimoma_artist = false_pos_artists[i][0].lower()
            rijks_ori_artist = false_pos_artists[i][1]
            rijks_artist = false_pos_artists[i][1].lower()
            for char in rijks_artist:
                if not char.isalpha():
                    #print char
                    substr = rijks_artist.split(char,1)[0]
                    break
                else:
                    substr = rijks_artist.split(char,1)[0]
            if wikimoma_ori_artist.find(substr) != -1:
                list.append([wikimoma_ori_artist, rijks_ori_artist])
    dd.io.save(os.path.join(data_folder, 'sub_artist_names_w_r.h5'), list)
    print 'list has saved!'
    return list


def update_artist_map_df(data_folder, df_first, df_second, same_artists_map_list, same_artist_list, base_name_first, base_name_second):
    """
    update artist names for the second df based on the first df
    Args:
        data_folder:folder for saving data
        df_first:the first dataframe
        df_second:the second dataframe
        same_artists_map_list:same artist map list
        same_artist_list:same artist name list
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'

    Returns:

    """
    update_artist_map = same_artist_list + same_artists_map_list
    file_name = 'same_artists_map_' + base_name_first + '_' + base_name_second + '.h5'
    dd.io.save(os.path.join(data_folder, file_name), update_artist_map)
    df_second = update_artist_names(data_folder, update_artist_map, df_first, df_second, base_name_second)
    return df_second


def check_update_artist_names(data_folder, base_name_first, df_first, base_name_second, df_second):
    """
    after checking false pos. and duplicates dataset, check if artist names are from the same artists
    Args:
        data_folder: folder for saving data
        base_name_first: e.g. 'wiki'
        df_first: the first dataframe
        base_name_second: e.g. 'moma'
        df_second: the second dataframe

    Returns: different artist names but from the same artists

    """
    artists_first, artists_second = get_artist_names(data_folder, base_name_first, df_first, base_name_second, df_second)
    substr_artists_list = check_substr(data_folder, artists_first, artists_second, base_name_first, base_name_second)
    substr_artists_list = del_false_artist_map(substr_artists_list)
    wikipedia_same_artists_list = check_wikipedia(data_folder, artists_first, artists_second, base_name_first, base_name_second)
    same_artists_map_list = substr_artists_list + wikipedia_same_artists_list
    # save same artist names map
    file_name = 'same_artists_map_' + base_name_first + '_' + base_name_second + '.h5'
    dd.io.save(os.path.join(data_folder, file_name), same_artists_map_list)
    update_artist_names(data_folder, same_artists_map_list, df_first, df_second, base_name_second)
    # check this list and find right artist pairs that are from the same artists
    return same_artists_map_list


def check_update_artist_names_sec(data_folder, base_name_first, df_first, base_name_second, df_second):
    """
    second method for checking and update same artist name list
    Args:
        data_folder:folder for saving data
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'
        df_first:the first dataframe
        df_second:the second dataframe

    Returns: same artist name list

    """
    artists_first, artists_second = get_artist_names_sec(data_folder, base_name_first, df_first, base_name_second, df_second)
    wikipedia_same_artists_list = check_wikipedia(data_folder, artists_first, artists_second, base_name_first, base_name_second)
    return wikipedia_same_artists_list


if __name__ == '__main__':
    # test
    data_folder = '/export/home/jli/workspace/data_after_run/'
    false_neg_artist_list = dd.io.load('/export/home/jli/workspace/data_after_run/false_pos_pairs_artists_wiki_moma.h5')
    modify_false_neg_artist_list(data_folder, false_neg_artist_list)
    df_first = pd.read_hdf(os.path.join(data_folder, 'wiki_info.hdf5'))
    df_second = pd.read_csv(os.path.join(data_folder, 'moma_info_filter_classification.csv'), index_col='id')
    same_artists_map_list = dd.io.load(os.path.join(data_folder, 'unique_same_artist_list_wiki_moma.h5'))
    same_artist_list = dd.io.load(os.path.join(data_folder, 'same_artists_map_wiki_moma.h5'))
    base_name_first = 'wiki'
    base_name_second = 'moma'
    update_artist_map_df(data_folder, df_first, df_second, same_artists_map_list, same_artist_list, base_name_first, base_name_second)