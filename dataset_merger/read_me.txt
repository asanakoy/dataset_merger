'run.py' is the main function for the whole process

data_folder = '/export/home/jli/workspace/readable_code_data'
result path after merging wiki, moma and rijks:/export/home/jli/workspace/data_after_run/info_wiki_moma_rijks_final_result.hdf5'

data_folder = '/export/home/jli/workspace/data_after_run'
img_folder_wiki = '/export/home/asanakoy/workspace/wikiart/images/'
img_folder_moma = '/export/home/jli/workspace/moma_boder_cropped/'
img_folder_rijks = '/export/home/jli/workspace/rijks_images/jpg2/'

For merging wiki and moma steps:
step 0:load two dataframes
step 1: filter classification for moma
step 2:extract path and save it
step 3:extract features for wiki and moma
step 4:compute cosine similarity
step 5:unify and upgrade artist names that from the same artists
step 6:split the second dataframe into duplicate part and unique part
step 7:check if there exists same artists in false_neg_artists list
step 8:upgrade artist map and dataframe based on same_artist_list
step 9:visualize false neg. pairs
step 10:split the second DataFrame into duplicate and unique parts and merge unique part with the first DataFrame
step 11:update wiki
step 12:merge wiki and moma_unique


For merging wiki_moma and rijks steps:
step 0:load dataframe
step 1: filter genre for rijks
step 2:extract path combination between first and second dataframe
step 3:extract features for wiki_moma and rijks
step 4:compute cosine similarity
step 5:unify and upgrade artist names that from the same artists
step 6:split the second dataframe into duplicate part and unique part
step 7:check subnames for artists
step 8:upgrade artist names based on subnames
step 9:false neg. pairs visualization
step 10:split rijks into unique part and duplicate part
step 11:update wikimoma info based on rijks duplicate info
step 12:merge wikimoma with rijks unique part
step 13:add source column



NOTE:
Jin:: 
For moma and rijks: I've already filter classification and genre for them.
For rijks and moma images: I've already cropped border for moma and convert RGB for rijks. And you can get image folder path through 'read_me.txt'
