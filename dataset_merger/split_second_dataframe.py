import deepdish as dd
import pandas as pd
import numpy as np
import collections
import os

def unique_part_second_dataframe(data_folder, moma, duplicates_list, base_name_second):
    """
    split moma into unique and duplicate parts
    Args:
        data_folder: folder for saving data
        moma: moma df
        duplicates_list: duplicate list for moma
        base_name_second: e.g. 'moma'

    Returns:unique part for moma

    """
    moma.drop({'Title', 'ArtistBio'}, 1, inplace=True)
    moma['is_duplicate'] = ""
    moma['is_duplicate'] = moma['is_duplicate'].astype(bool)
    moma['image_id'] = moma.index.values
    moma = moma.rename(columns = {'Artist':'artist_slug', 'Date': 'date', 'URL':'page_url', 'Medium':'media',
                                  'Dimensions':'dimensions', 'CreditLine':'credit_line', 'MoMANumber':'moma_number',
                                  'Classification':'classification', 'Department':'department', 'DateAcquired':'date_acquired',
                                  'CuratorApproved':'curator_approved'})

    moma_new_dup_list = []
    for i in range(len(duplicates_list)):
        moma_new_dup_list.append(duplicates_list[:,1][i].astype(int))
    moma_new_dup_list = np.unique(np.sort(np.asarray(moma_new_dup_list)))
    moma_unique = moma.ix[~moma.image_id.isin(moma_new_dup_list)]
    keep_list = ['Photograph', 'Drawing', 'Illustrated Book', 'Painting', 'Collage']
    moma_unique = moma_unique[moma_unique.classification.isin(keep_list)]
    print moma_unique.classification.value_counts()
    print len(moma_unique)
    # save moma_unique
    save_moma_unique_name = base_name_second + '_info_unique' + '.csv'
    if not os.path.exists(os.path.join(data_folder, save_moma_unique_name)):
        moma_unique.to_csv(os.path.join(data_folder, save_moma_unique_name), index='image_id')
        print 'save unique part for csv'
    return moma_unique


def merge_first_second_df(data_folder, df_first, df_second_unique, base_name_first, base_name_second):
    """
    merge the first and second df
    Args:
        data_folder: folder for saving data
        df_first: the first df
        df_second_unique: unique part for the second df
        base_name_first: e.g.'wiki'
        base_name_second: e.g.'moma'

    Returns:df after merging

    """
    save_name = 'info_' + base_name_first + '_' + base_name_second + '_merged' + '.hdf5'
    save_path = os.path.join(data_folder, save_name)
    #df_first = pd.read_hdf(df_first)
    df_first['image_id'] = df_first['image_id'].apply(lambda s: s.decode('utf-8') if isinstance(s, str) else s)
    #df_second_unique = pd.read_csv(df_second_unique, encoding='utf-8', index_col='id')
    print df_first.columns
    print df_second_unique.columns
    print len(df_first)
    print len(df_second_unique)
    joined = pd.concat([df_first, df_second_unique], ignore_index=False)
    print joined.columns
    print len(joined)
    joined.to_hdf(save_path, 'image_id', format='fixed')
    return joined


def get_unique_rijks(data_folder, rijks, duplicates):
    """
    split rijks into unique and duplicate parts
    Args:
        data_folder:folder for saving data
        rijks:rijks df
        duplicates:duplicate list

    Returns:unique and duplicate part

    """
    print rijks.columns
    print len(rijks)
    print len(duplicates)
    rijks_dup_list = []
    for i in range(len(duplicates)):
        rijks_dup_list.append(duplicates[i][1])
    print len(rijks_dup_list)
    # print duplicates in the list
    print [item for item, count in collections.Counter(rijks_dup_list).items() if count > 1]
    rijks_unique_info = rijks[~rijks.image_id.isin(rijks_dup_list)]
    rijks_dup_info = rijks[rijks.image_id.isin(rijks_dup_list)]
    # save rijks_dup_info.hdf5 and rijks_unique_info.hdf5
    rijks_unique_info.to_hdf(os.path.join(data_folder, 'rijks_unique_info.hdf5'), 'image_id')
    rijks_dup_info.to_hdf(os.path.join(data_folder, 'rijks_dup_info.hdf5'), 'image_id')
    return rijks_unique_info, rijks_dup_info


if __name__ == '__main__':
    # test
    data_folder = '/export/home/jli/workspace/readable_code_data/'
    df_second = pd.read_csv('/export/home/jli/workspace/readable_code_data/moma_info_unique.csv', encoding='utf-8', index_col='id')
    df_first = pd.read_hdf('/export/home/jli/workspace/readable_code_data/wiki_info_update.hdf5')
    #duplicates_list = dd.io.load('/export/home/jli/workspace/readable_code_data/split_dup_uniq_info/duplicates_wiki_moma.h5')
    #moma_unique = unique_part_second_dataframe(data_folder, df_second, duplicates_list, 'moma')
    merge_first_second_df(data_folder, df_first, df_second, 'wiki', 'moma')
    #rijks = pd.read_hdf(os.path.join(data_folder, 'rijks_info_after_unify_artist_names.hdf5'))
    #duplicates = dd.io.load('/export/home/jli/workspace/readable_code_data/split_info_wikimoma_rijks/duplicates_wikimoma_rijks_after_substr_detect.h5')
    #get_unique_rijks(data_folder, rijks, duplicates)