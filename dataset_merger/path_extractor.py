import deepdish as dd
import pandas as pd
import os

def path_extractor(data_folder, base_name_first, df_first, img_folder_first, base_name_second, df_second, img_folder_second):
    """
    extract img path for df
    Args:
        data_folder:folder for saving data
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'
        df_first: the first dataframe
        img_folder_first: img folder for the first art gallery
        df_second: the second dataframe
        img_folder_second: img folder for the second art gallery

    Returns:img path for the first df and second df

    """
    path_list_first = []
    path_list_second = []
    num_works_first = len(df_first)
    num_works_second = len(df_second)

    for i in range(num_works_first):
        wiki_name = df_first.image_id.index[i] + '.jpg'
        save_wiki_name = os.path.join(img_folder_first, wiki_name)
        path_list_first.append(save_wiki_name)
    for i in range(num_works_second):
        moma_name = str(df_second.index[i]) + '.jpg'
        save_moma_name = os.path.join(img_folder_second, moma_name)
        path_list_second.append(save_moma_name)
    # save file
    save_file_name = []
    save_file_name.append('img_path_list_' + base_name_first + '.h5')
    save_file_name.append('img_path_list_' + base_name_second + '.h5')
    dd.io.save(os.path.join(data_folder, save_file_name[0]), path_list_first)
    dd.io.save(os.path.join(data_folder, save_file_name[1]), path_list_second)
    print 'save image path list successfully'
    return path_list_first, path_list_second


def first_second_unique_path_extractor(data_folder, img_folder_first, img_folder_second, df_first, df_second_unique, base_name_first, \
                                       base_name_second):
    """
    path after merging wiki and moma
    Args:
        data_folder: folder for saving data
        img_folder_first: img folder for the first art gallery
        img_folder_second: img folder for the second art gallery
        df_first: the first dataframe
        df_second_unique: the second unique dataframe
        base_name_first: e.g. 'wiki'
        base_name_second:e.g. 'moma'

    Returns:

    """
    combine_path = []
    for i in range(len(df_first)):
        wiki_img_name = str(df_first.image_id.values[i]) + '.jpg'
        combine_path.append(os.path.join(img_folder_first, wiki_img_name))
    for j in range(len(df_second_unique)):
        moma_img_name = str(df_second_unique.image_id.values[j]) + '.jpg'
        combine_path.append(os.path.join(img_folder_second, moma_img_name))
    file_name = 'img_path_' + base_name_first + base_name_second + '.h5'
    dd.io.save(os.path.join(data_folder, file_name), combine_path)
    return combine_path


def path_list_extactor_all(data_folder, rijks, img_folder, base_name):
    """

    Args:
        data_folder: folder for saving data
        rijks: dataframe
        img_folder: rijks image folder
        base_name: e.g. 'rijks'

    Returns:path for rijks data

    """
    rijks_path_list = []
    num_works_rijks = len(rijks)
    rijks = (rijks.image_id.values).tolist()
    for i in range(num_works_rijks):
        rijks_name = str(rijks[i]) + '.jpg'
        save_rijks_name = os.path.join(img_folder, rijks_name)
        rijks_path_list.append(save_rijks_name)
    file_name = 'img_path_list_' + base_name + '.h5'
    dd.io.save(os.path.join(data_folder, file_name), rijks_path_list)
    print 'rijks path list has saved!'
    return rijks_path_list


if __name__ == '__main__':
    # test
    data_folder = '/export/home/jli/workspace/readable_code_data'
    img_folder_first = '/export/home/asanakoy/workspace/wikiart/images/'
    img_folder_second = '/export/home/jli/workspace/moma_boder_cropped/'
    img_folder_third = '/export/home/jli/workspace/rijks_images/jpg2/'
    wiki = pd.read_hdf('/export/home/jli/workspace/readable_code_data/wiki_info.hdf5')
    unique_moma = pd.read_csv('/export/home/jli/workspace/readable_code_data/moma_info_unique.csv', index_col='id')
    #combine_path = first_second_unique_path_extractor(data_folder, img_folder_first, img_folder_second, wiki, unique_moma, 'wiki', 'moma')
    rijks = pd.read_hdf('/export/home/jli/workspace/readable_code_data/rijks_info.hdf5')
    rijks_path_list = path_list_extactor_all(data_folder, rijks, img_folder_third, 'rijks')