from path_extractor import path_extractor, first_second_unique_path_extractor, path_list_extactor_all
from feature_extractor import feature_extractor
from compute_similarity import compute_similarity
from check_update_artist_names import check_update_artist_names, modify_false_neg_artist_list, update_artist_map_df, check_update_artist_names_sec, detect_sub_artist_names
from split_duplicates_unique import split_duplicates_unique, split_duplicates_unique_sec
from false_neg_pairs_visualization import false_neg_pairs_visualization
from split_second_dataframe import unique_part_second_dataframe, merge_first_second_df, get_unique_rijks
from update_dataframe import update_wiki, update_same_artist_names, update_w_m_based_rijks_dup, merge_wikimoma_rijks, add_source_column, filter_genre
import pandas as pd
import os

snapshot_path_first = '/export/home/jli/PycharmProjects/practical/data/bvlc_alexnet.tf'
snapshot_path_second = '/export/home/asanakoy/workspace/wikiart/cnn/artist_50/rs_balance/model/snap_iter_335029.tf'
mean_path_second = '/export/home/asanakoy/workspace/wikiart/cnn/artist_50/rs_balance/data/mean.npy'
wiki_img_folder_path = '/export/home/asanakoy/workspace/wikiart/images/'
moma_img_folder_path = '/export/home/asanakoy/workspace/moma/images/1_filtered/'


def run(data_folder, base_name_first, img_folder_first, df_first_name, base_name_second, img_folder_second,\
        df_second_name, img_folder_third, df_third_name, base_name_third):
    ### merge the first and second dataframes
    # step 0:load two dataframe
    df_first = pd.read_hdf(os.path.join(data_folder, df_first_name))
    df_second = pd.read_csv(os.path.join(data_folder, df_second_name), index_col='id')
    # step 1:extract path and save it
    img_path_first, img_path_second = path_extractor(data_folder, base_name_first, df_first, img_folder_first, base_name_second, df_second, img_folder_second)
    # step 2:extract features for wiki and moma
    features_first = feature_extractor(snapshot_path_second, img_path_first, data_folder, base_name_first, mean_path=mean_path_second)
    features_second = feature_extractor(snapshot_path_second, img_path_second, data_folder, base_name_second, mean_path=mean_path_second)
    # step 3:compute cosine similarity
    similarity_matrix = compute_similarity(data_folder, features_first, features_second, base_name_first, base_name_second)
    # step 4:unify and upgrade artist names that from the same artists
    same_artists_map_list = check_update_artist_names(data_folder, base_name_first, df_first, base_name_second, df_second)
    # step 5:split the second dataframe into duplicate part and unique part
    false_neg_pairs_name, false_neg_artists, duplicates_list = split_duplicates_unique(data_folder, similarity_matrix, df_first, df_second, img_path_first, img_path_second, base_name_first, base_name_second)
    # step 6:check if there exists same artists in false_neg_artists list
    same_artist_list = modify_false_neg_artist_list(false_neg_artists)
    # step 7:upgrade artist map and dataframe based on same_artist_list
    update_artist_map_df(data_folder, df_first, df_second, same_artists_map_list, same_artist_list, 'wiki', 'moma')
    # step 8:visualize false neg. pairs
    false_neg_pairs_visualization(wiki_img_folder_path, moma_img_folder_path, data_folder, false_neg_pairs_name, false_neg_artists)
    # step 9:update wiki
    df_first = update_wiki(data_folder, df_first, df_second, duplicates_list)
    # step 10:split the second DataFrame into duplicate and unique parts and merge unique part with the first DataFrame
    df_second_unique = unique_part_second_dataframe(data_folder, df_second, duplicates_list, base_name_second)
    df_joined_fir_sec = merge_first_second_df(data_folder, df_first, df_second_unique, base_name_first, base_name_second)

    ### merge rijks with wiki and moma
    # step 0:load dataframe
    df_third = pd.read_hdf(os.path.join(data_folder, df_third_name))
    # step 1:extract path combination between first and second dataframe
    combine_fir_sec_path = first_second_unique_path_extractor(data_folder, img_folder_first, img_folder_second, df_first, df_second_unique, base_name_first, \
                                       base_name_second)
    # step 2:extract features for wiki_moma and rijks
    base_name_fir_sec = base_name_first + base_name_second
    img_path_third = path_list_extactor_all(data_folder, df_third, img_folder_third, base_name_third)
    features_combine_fir_sec = feature_extractor(snapshot_path_second, combine_fir_sec_path, data_folder, base_name_fir_sec, mean_path=mean_path_second)
    features_third = feature_extractor(snapshot_path_second, img_path_third, data_folder, base_name_third, mean_path=mean_path_second)
    # step 3:compute cosine similarity
    similarity_matrix_sec = compute_similarity(data_folder, features_combine_fir_sec, features_third, base_name_fir_sec, base_name_third)
    # step 4:unify and upgrade artist names that from the same artists
    wikipedia_list = check_update_artist_names_sec(data_folder, base_name_fir_sec, df_joined_fir_sec, base_name_third, df_third)
    # step 5:split the second dataframe into duplicate part and unique part
    false_neg_pairs_name_sec, false_neg_artists_sec, duplicates_list_sec = split_duplicates_unique_sec(data_folder, similarity_matrix_sec, combine_fir_sec_path, img_path_third, df_joined_fir_sec, df_third, df_first,base_name_fir_sec, base_name_third)
    # step 6:check subnames for artists
    subname_list = detect_sub_artist_names(false_neg_artists_sec)
    # step 7:upgrade artist names based on subnames
    df_third = update_same_artist_names(data_folder, wikipedia_list, subname_list, df_third)
    # step 8:split rijks into unique part and duplicate part
    unique_part, dup_part = get_unique_rijks(data_folder, df_third, duplicates_list_sec)
    # step 9:filter some genres
    unique_part = filter_genre(data_folder, unique_part)
    # step 10:update wikimoma info based on rijks duplicate info
    wikimoma_update = update_w_m_based_rijks_dup(data_folder, df_third, df_joined_fir_sec, dup_part)
    # step 11:merge wikimoma with rijks unique part
    df_wikimoma_rijks = merge_wikimoma_rijks(data_folder, wikimoma_update, df_third)
    # step 12:add source column
    add_source_column(data_folder, df_first, df_second_unique, unique_part, df_wikimoma_rijks)