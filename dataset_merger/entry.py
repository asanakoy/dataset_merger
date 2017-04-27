from readable_codes.run import run

data_folder = '/export/home/jli/workspace/readable_code_data'
img_folder_first = '/export/home/asanakoy/workspace/wikiart/images/'
img_folder_second = '/export/home/jli/workspace/moma_boder_cropped/'
img_folder_third = '/export/home/jli/workspace/rijks_images/jpg2/'

df_first_name = 'wiki_info.hdf5'
df_second_name = 'moma_info.csv'
df_third_name = 'rijks_info.hdf5'
base_name_first = 'wiki'
base_name_second = 'moma'
base_name_third = 'rijks'

run(data_folder, base_name_first, img_folder_first, df_first_name, base_name_second, img_folder_second, df_second_name,\
    img_folder_third, df_third_name, base_name_third)