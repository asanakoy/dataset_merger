import pandas as pd
import os
import time

from feature_extractor import feature_extractor

from path_extractor import path_extractor, first_second_unique_path_extractor, path_list_extactor_all
from compute_similarity import compute_similarity
from check_update_artist_names import check_update_artist_names, modify_false_neg_artist_list, update_artist_map_df, check_update_artist_names_sec, detect_sub_artist_names
from split_duplicates_unique import split_duplicates_unique, split_duplicates_unique_sec
from false_neg_pairs_visualization import false_neg_pairs_visualization
from split_second_dataframe import unique_part_second_dataframe, merge_first_second_df, get_unique_rijks, moma_classification_filter
from update_dataframe import update_wiki, update_same_artist_names, update_w_m_based_rijks_dup, merge_wikimoma_rijks, add_source_column, filter_genre


data_folder = '/export/home/jli/workspace/data_after_run/'
img_folder_first = '/export/home/asanakoy/workspace/wikiart/images/'
img_folder_second = '/export/home/jli/workspace/moma_boder_cropped/'
img_folder_third = '/export/home/jli/workspace/rijks_images/jpg2/'

df_first_name = 'wiki_info.hdf5'
df_second_name = 'moma_info.csv'
df_third_name = 'rijks_info.hdf5'
base_name_first = 'wiki'
base_name_second = 'moma'
base_name_third = 'rijks'

snapshot_path_first = '/export/home/jli/PycharmProjects/practical/data/bvlc_alexnet.tf'
snapshot_path_second = '/export/home/asanakoy/workspace/wikiart/cnn/artist_50/rs_balance/model/snap_iter_335029.tf'
mean_path_second = '/export/home/asanakoy/workspace/wikiart/cnn/artist_50/rs_balance/data/mean.npy'


def run(data_folder, base_name_first, img_folder_first, df_first_name, base_name_second,
        img_folder_second, df_second_name, img_folder_third, df_third_name, base_name_third):
    ### merge the first and second dataframes
    # step 0:load two dataframe
    df_first = pd.read_hdf(os.path.join(data_folder, df_first_name))
    df_second = pd.read_csv(os.path.join(data_folder, df_second_name), index_col='id')
    # step 1: filter classification for moma
    df_second = moma_classification_filter(data_folder, df_second)
    # step 2:extract path and save it
    img_path_first, img_path_second = path_extractor(data_folder, base_name_first, df_first,
                                                     img_folder_first, base_name_second,
                                                     df_second, img_folder_second)
    # step 3:extract features for wiki and moma

    features_second = feature_extractor(snapshot_path_second, img_path_second, data_folder,
                                        base_name_second, mean_path=mean_path_second)
    features_first = feature_extractor(snapshot_path_second, img_path_first, data_folder,
                                       base_name_first, mean_path=mean_path_second)
    # step 4:compute cosine similarity
    similarity_matrix = compute_similarity(data_folder, features_first, features_second,
                                           base_name_first, base_name_second)
    # step 5:unify and upgrade artist names that from the same artists
    same_artists_map_list = check_update_artist_names(data_folder, base_name_first, df_first,
                                                      base_name_second, df_second)
    # step 6:split the second dataframe into duplicate part and unique part
    false_neg_pairs_name, false_neg_artists, duplicates_list = \
        split_duplicates_unique(data_folder, similarity_matrix, df_first, df_second,
                                img_path_first, img_path_second, base_name_first, base_name_second)
    # step 7:check if there exists same artists in false_neg_artists list
    same_artist_list = modify_false_neg_artist_list(data_folder, false_neg_artists)
    # step 8:upgrade artist map and dataframe based on same_artist_list
    df_second = update_artist_map_df(data_folder, df_first, df_second, same_artists_map_list,
                                     same_artist_list, 'wiki', 'moma')

    # step 9:visualize false neg. pairs
    if not os.path.exists(os.path.join(data_folder, 'visualization_wiki_moma')):
        os.mkdir(os.path.join(data_folder, 'visualization_wiki_moma'))
    visualization_folder = os.path.join(data_folder, 'visualization_wiki_moma')
    false_neg_pairs_name, false_neg_artists, duplicates_list = \
        split_duplicates_unique(data_folder, similarity_matrix, df_first, df_second,
                                img_path_first, img_path_second, base_name_first, base_name_second)
    false_neg_pairs_visualization(img_folder_first, img_folder_second, visualization_folder,
                                  false_neg_pairs_name, false_neg_artists)

    # step 10:split the second DataFrame into duplicate and unique parts and merge unique part with the first DataFrame
    df_second_unique, moma_dup = unique_part_second_dataframe(data_folder, df_second,
                                                              duplicates_list, base_name_second)
    # step 11:update wiki
    df_first = update_wiki(data_folder, df_first, moma_dup, duplicates_list)
    # step 12:merge wiki and moma_unique
    df_joined_fir_sec = merge_first_second_df(data_folder, df_first, df_second_unique,
                                              base_name_first, base_name_second)

    ### merge rijks with wiki and moma
    # step 0:load dataframe
    df_third = pd.read_hdf(os.path.join(data_folder, df_third_name))
    # step 1: filter genre for rijks
    df_third = filter_genre(data_folder, df_third)
    # step 2:extract path combination between first and second dataframe
    img_path_fir_sec = first_second_unique_path_extractor(data_folder, img_folder_first,
                                                          img_folder_second, df_first, df_second_unique,
                                                          base_name_first,base_name_second)
    img_path_third = path_list_extactor_all(data_folder, df_third, img_folder_third, base_name_third)
    # step 3:extract features for wiki_moma and rijks
    base_name_fir_sec = base_name_first + base_name_second
    features_combine_fir_sec = feature_extractor(snapshot_path_second, img_path_fir_sec,
                                                 data_folder, base_name_fir_sec,
                                                 mean_path=mean_path_second)
    features_third = feature_extractor(snapshot_path_second, img_path_third, data_folder,
                                       base_name_third, mean_path=mean_path_second)
    # step 4:compute cosine similarity
    similarity_matrix_sec = compute_similarity(data_folder, features_combine_fir_sec,
                                               features_third, base_name_fir_sec, base_name_third)
    # step 5:unify and upgrade artist names that from the same artists
    wikipedia_list = check_update_artist_names_sec(data_folder, base_name_fir_sec,
                                                   df_joined_fir_sec, base_name_third, df_third)

    # step 6:split the second dataframe into duplicate part and unique part
    similarity_matrix_sec_path = os.path.join(data_folder, 'similarity_wikimoma_rijks.h5')
    false_neg_pairs_name_sec, false_neg_artists_sec, duplicates_list_sec = \
        split_duplicates_unique_sec(data_folder, similarity_matrix_sec_path, img_path_fir_sec,
                                    img_path_third, df_joined_fir_sec, df_third, df_first,
                                    base_name_fir_sec, base_name_third)

    # step 7:check subnames for artists
    subname_list = detect_sub_artist_names(data_folder, false_neg_artists_sec)
    # step 8:upgrade artist names based on subnames
    df_third = update_same_artist_names(data_folder, wikipedia_list, subname_list, df_third)

    # step 9:false neg. pairs visualization
    false_neg_pairs_name_sec, false_neg_artists_sec, duplicates_list_sec = \
        split_duplicates_unique_sec(data_folder, similarity_matrix_sec_path, img_path_fir_sec,
                                    img_path_third, df_joined_fir_sec, df_third, df_first,
                                    base_name_fir_sec, base_name_third)
    if not os.path.exists(os.path.join(data_folder, 'visualization_wikimoma_rijks')):
        os.mkdir(os.path.join(data_folder, 'visualization_wikimoma_rijks'))
    sub_visual_folder_sec = os.path.join(data_folder, 'visualization_wikimoma_rijks')
    false_neg_pairs_visualization(img_path_fir_sec, img_path_third, sub_visual_folder_sec,
                                  false_neg_pairs_name_sec, false_neg_artists_sec)

    # step 10:split rijks into unique part and duplicate part
    unique_part, dup_part = get_unique_rijks(data_folder, df_third, duplicates_list_sec)
    # step 11:update wikimoma info based on rijks duplicate info
    wikimoma_update = update_w_m_based_rijks_dup(data_folder, dup_part,
                                                 df_joined_fir_sec, duplicates_list_sec)
    # step 12:merge wikimoma with rijks unique part
    df_wikimoma_rijks = merge_wikimoma_rijks(data_folder, wikimoma_update, unique_part)
    # step 13:add source column
    add_source_column(data_folder, df_first, df_second_unique, df_wikimoma_rijks)


if __name__ == '__main__':
    time_start = time.time()
    run(data_folder, base_name_first, img_folder_first, df_first_name, base_name_second,
        img_folder_second, df_second_name, img_folder_third, df_third_name, base_name_third)
    print 'Elapsed time: {:.2f} sec'.format(time.time() - time_start)
