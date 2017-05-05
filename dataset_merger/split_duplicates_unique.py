from __future__ import division
import numpy as np
import pandas as pd
import deepdish as dd
import time
import sys
import os


def split_duplicates_unique(data_folder, similarity_matrix, df_first, df_second, img_path_first, img_path_second, base_name_first, base_name_second):
    """
    for wiki and moma, compute top similarity and check if they are from the same artists
    Args:
        data_folder: folder for saving data
        similarity_matrix: similarity matrix
        df_first: the first dataframe
        df_second: the second dataframe
        img_path_first: img path for the first art gallery
        img_path_second: img path for the second art gallery
        base_name_first: e.g. 'wiki'
        base_name_second: e.g. 'moma'

    Returns:
        false_pos_pairs_path_wikimoma_rijks
        false_pos_pairs_artist_list
        duplicate_image_names: duplicates (wikimoma, rijks) : list of basenames of duplicate pairs of image

    """
    threshold = 0.98
    duplicate_image_names = []
    false_pos_pairs_artist_list = []
    false_pos_pairs_path_first_second = []
    # find top scoring from similarities
    start = time.time()
    num_samples = len(similarity_matrix)
    if not os.path.exists(os.path.join(data_folder, 'match_position_list_wiki_moma.h5')):
        match_position = []
        for loc_wiki in range(num_samples):
            top_scoring = np.max(similarity_matrix[loc_wiki])
            if top_scoring >= threshold:
                loc_moma = np.where(similarity_matrix[loc_wiki] == top_scoring)[0][0]
                # matching positions between wiki and moma
                match_position.append([loc_wiki, loc_moma])
        dd.io.save(os.path.join(data_folder, 'match_position_list_wiki_moma.h5'), match_position)
        print "match_position_list.h5 has saved!"
    else:
        match_position = dd.io.load(os.path.join(data_folder, 'match_position_list_wiki_moma.h5'))
    num_pairs = len(match_position)
    sum_error = 0
    for i in range(num_pairs):
        loc_wiki = match_position[i][0]
        loc_moma = match_position[i][1]
        # search for its corresponding artist
        base_wiki = os.path.basename(img_path_first[loc_wiki]).split('.', 1)[0]
        base_moma = os.path.basename(img_path_second[loc_moma]).split('.', 1)[0]
        artist_wiki = df_first[df_first.image_id == base_wiki].artist_slug[0]
        artist_moma = df_second[df_second.index == int(base_moma)].Artist.values[0]
        if str(artist_wiki) == artist_moma:
            duplicate_image_names.append([base_wiki, base_moma])
        else:
            sum_error += 1
            # print error pairs
            error_match = [str(artist_wiki), artist_moma]
            false_pos_pairs_artist_list.append(error_match)
            # save its path
            wiki_name = base_wiki + '.jpg'
            moma_name = base_moma + '.jpg'
            false_pos_pairs_path_first_second.append([wiki_name, moma_name])
    assert sum_error <= num_pairs

    duplicate_image_names = np.array(duplicate_image_names, dtype=str)
    duplicate_name = 'duplicates_' + base_name_first + '_' + base_name_second + '.h5'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    dd.io.save(os.path.join(data_folder, duplicate_name), duplicate_image_names)

    # save error pairs and its paths into h5
    h5_artist_name = 'false_pos_pairs_artists_' + base_name_first + '_' + base_name_second + '.h5'
    h5_path_name = 'false_pos_pairs_path_' + base_name_first + '_' + base_name_second + '.h5'
    false_pos_pairs_artist_list = np.array(false_pos_pairs_artist_list, dtype=str)
    false_pos_pairs_path_first_second = np.array(false_pos_pairs_path_first_second, dtype=str)
    dd.io.save(os.path.join(data_folder, h5_path_name), false_pos_pairs_path_first_second)
    dd.io.save(os.path.join(data_folder, h5_artist_name), false_pos_pairs_artist_list)
    print "threshold is:", threshold
    print "there are " + str(num_pairs - sum_error) + " true pos. pairs from " + str(num_samples) + " pairs in total."
    print "number of false pos pairs: ", sum_error
    print "error rate:", sum_error / num_pairs if num_pairs else 0
    # save log into txt
    txt_name = 'log_after_unify_name_final_threshold_' + str(threshold) + '.txt'
    with open(os.path.join(data_folder, txt_name), "w") as text_file:
        text_file.write("threshold is: %f\n" % threshold)
        text_file.write("there are true %d pos. pairs from %d pairs in total.\n" % (num_pairs - sum_error, num_samples))
        text_file.write("number of false pos pairs: %f\n" % sum_error)
        text_file.write("false neg. rate: %f\n" % (sum_error / num_pairs if num_pairs else 0))
    end_whole = time.time()
    elapsed_whole = end_whole - start
    print "it took {} minutes to finish whole splitting process.".format(elapsed_whole / 60)
    return false_pos_pairs_path_first_second, false_pos_pairs_artist_list, duplicate_image_names


def split_duplicates_unique_sec(data_folder, similarity_matrix_path, path_wikimoma, path_rijks, wikimoma, rijks, wiki, base_name_first, base_name_second):
    """
    for rijks and wiki_moma, compute top similarity and check if they are from the same artists
    Args:
        data_folder: folder for saving data
        similarity_matrix:similarity matrix
        path_wikimoma:wiki_moma img path after merging
        path_rijks:img path for rijks
        wikimoma:df after merging wiki and moma
        rijks:rijks df
        wiki:wiki df
        base_name_first: e.g. 'wiki'
        base_name_second: e.g. 'moma'

    Returns:false_pos_pairs_path_wikimoma_rijks ,false_pos_pairs_artist_list, duplicates_wikimoma_rijks

    """
    print 'begin'
    final_folder_name = 'split_info_wikimoma_rijks'
    save_folder_path = os.path.join(data_folder, final_folder_name)
    if not os.path.exists(os.path.join(data_folder, save_folder_path)):
        os.mkdir(save_folder_path)
    save_folder_path = os.path.join(data_folder, save_folder_path)
    threshold = 0.98
    duplicates_wikimoma_rijks = []
    false_pos_pairs_artist_list = []
    false_pos_pairs_path_wikimoma_rijks = []
    num_wiki = len(wiki)
    print os.path.join(save_folder_path, 'match_position_list_wm_rijks.h5')
    if not os.path.exists(os.path.join(save_folder_path, 'match_position_list_wm_rijks.h5')):
        # find top scoring from similarities
        print 'computing'
        start = time.time()
        similarity_matrix = dd.io.load(similarity_matrix_path)
        end = time.time()
        elapsed = end - start
        print "it took {} minutes to load sim matrix".format(elapsed / 60)
        num_samples = len(similarity_matrix)
        match_position = []
        for loc_wikimoma in range(num_samples):
            top_scoring = np.max(similarity_matrix[loc_wikimoma])
            if top_scoring >= threshold:
                loc_rijks = np.where(similarity_matrix[loc_wikimoma] == top_scoring)[0][0]
                # matching positions between wiki and moma
                #print [loc_wiki, loc_moma]
                match_position.append([loc_wikimoma, loc_rijks])
                # save this top scoring position
        dd.io.save(os.path.join(save_folder_path, 'match_position_list_wm_rijks.h5'), match_position)
        print "match_position_list.h5 has saved!"
    else:
        print 'prepare for loading match_position matrix'
        match_position = dd.io.load(os.path.join(save_folder_path, 'match_position_list_wm_rijks.h5'))
        #match_position = dd.io.load('/export/home/jli/workspace/readable_code_data/split_info_wikimoma_rijks/match_position_list.h5')
        print 'match_position.h5 has loaded!'
    num_pairs = len(match_position)
    sum_error = 0
    for i in range(num_pairs):
        sys.stdout.write("\r%d/%d" % (i, num_pairs))
        sys.stdout.flush()
        loc_wikimoma = match_position[i][0]
        loc_rijks = match_position[i][1]
        # search for its corresponding artist
        wikimoma_img_full_path = path_wikimoma[loc_wikimoma]
        rijks_img_full_path = path_rijks[loc_rijks]
        base_wikimoma = os.path.basename(wikimoma_img_full_path).split(".jpg",1)[0]##
        if loc_wikimoma >= num_wiki:
            base_wikimoma = int(base_wikimoma)
        base_rijks = os.path.basename(rijks_img_full_path).split(".jpg",1)[0]
        artist_wikimoma = wikimoma[wikimoma.image_id == base_wikimoma].artist_slug.values[0]
        artist_rijks = rijks[rijks.image_id == base_rijks].artist_slug.values[0]
        #print [artist_wikimoma, artist_rijks]
        if str(artist_wikimoma) == artist_rijks:
            duplicates_wikimoma_rijks.append([base_wikimoma, base_rijks])
        if not str(artist_wikimoma) == artist_rijks:
            sum_error += 1
            # print error pairs
            error_match = [str(artist_wikimoma), artist_rijks]
            false_pos_pairs_artist_list.append(error_match)
            # save its path
            false_pos_pairs_path_wikimoma_rijks.append([wikimoma_img_full_path, rijks_img_full_path])
    print "there are:", len(false_pos_pairs_path_wikimoma_rijks)
    duplicates_wikimoma_rijks = np.array(duplicates_wikimoma_rijks, dtype=unicode)
    duplicate_name = 'duplicates_wikimoma_rijks_after_substr_detect.h5'
    dd.io.save(os.path.join(save_folder_path, duplicate_name), duplicates_wikimoma_rijks)

    # save error pairs and its paths into h5
    h5_artist_name = 'false_pos_pairs_artists_' + base_name_first + '_' + base_name_second + '.h5'
    h5_path_name = 'false_pos_pairs_path_' + base_name_first + '_' + base_name_second + '.h5'
    false_pos_pairs_artist_list = np.array(false_pos_pairs_artist_list, dtype=unicode)
    false_pos_pairs_path_wikimoma_rijks = np.array(false_pos_pairs_path_wikimoma_rijks, dtype=unicode)
    dd.io.save(os.path.join(save_folder_path, h5_path_name), false_pos_pairs_path_wikimoma_rijks)
    dd.io.save(os.path.join(save_folder_path, h5_artist_name), false_pos_pairs_artist_list)

    print "threshold is:", threshold
    print "there are", num_pairs - sum_error, "pos. pairs from", num_pairs, "pairs in total."
    print "number of false pos. pairs:", sum_error
    error_rate = float(sum_error) / num_pairs if num_pairs else 0
    print "false pos. rate:", error_rate
    # save log into txt
    txt_name = 'log_wikimoma_rijks_after_substr_detect.txt'
    with open(os.path.join(save_folder_path, txt_name), "w") as text_file:
        text_file.write("threshold is: %f\n" % threshold)
        text_file.write("there are %d right pairs from %d pairs in total.\n" %
                        (num_pairs - sum_error, num_pairs))
        text_file.write("number of false pos. pairs: %f\n" % sum_error)
        text_file.write("false pos. rate: %f\n" % error_rate)
    return false_pos_pairs_path_wikimoma_rijks, false_pos_pairs_artist_list, duplicates_wikimoma_rijks


if __name__ == '__main__':
    # test
    similarity_path = '/export/home/jli/workspace/art_project/practical/data/similarity_wiki_moma.h5'
    start = time.time()
    #similarity_matrix = dd.io.load(similarity_path)
    end = time.time()
    elapsed = end - start
    #print "it takes " + `elapsed / 60` + " minutes to load similarity matrix."
    data_folder = '/export/home/jli/workspace/readable_code_data/'
    df_first = pd.read_hdf('/export/home/jli/workspace/readable_code_data/wiki_info.hdf5')
    df_second = pd.read_csv('/export/home/jli/workspace/readable_code_data/moma_update_artist_names.csv', index_col='id')
    img_path_first = dd.io.load('/export/home/jli/workspace/readable_code_data/img_path_list_wiki.h5')
    img_path_second = dd.io.load('/export/home/jli/workspace/readable_code_data/img_path_list_moma.h5')
    base_name_first = 'wiki'
    base_name_second = 'moma'
    #split_duplicates_unique(data_folder, similarity_matrix, df_first, df_second, img_path_first, img_path_second, base_name_first, base_name_second)
    #start = time.time()
    #similarity_matrix = dd.io.load('/export/home/jli/workspace/art_project/practical/data/final/similarity_rijks_wikimoma.h5')
    #end = time.time()
    #elapsed = end - start
    #print "it takes " + `elapsed / 60` + " minutes to load similarity matrix."
    path_wikimoma = dd.io.load('/export/home/jli/workspace/readable_code_data/path_combine_wiki_moma.h5')
    path_rijks = dd.io.load('/export/home/jli/workspace/readable_code_data/img_path_list_rijks.h5')
    wikimoma = pd.read_hdf('/export/home/jli/workspace/readable_code_data/info_wiki_moma_merged.hdf5')
    rijks = pd.read_hdf('/export/home/jli/workspace/readable_code_data/rijks_info_after_unify_artist_names.hdf5')
    split_duplicates_unique_sec(data_folder, '', path_wikimoma, path_rijks, wikimoma, rijks, df_first, 'wikimoma', 'rijks')