'entry.py' is the entry of the whole program
'run.py' is the main function for the whole process

data_folder = '/export/home/jli/workspace/readable_code_data'
result path after merging wiki, moma and rijks:'/export/home/jli/workspace/readable_code_data/info_wiki_moma_rijks_final_result.hdf5'

data_folder = '/export/home/jli/workspace/readable_code_data'
img_folder_wiki = '/export/home/asanakoy/workspace/wikiart/images/'
img_folder_moma = '/export/home/jli/workspace/moma_boder_cropped/'
img_folder_rijks = '/export/home/jli/workspace/rijks_images/jpg2/'

For merging wiki and moma steps:
step 0:load two dataframes
step 1:extract path and save it
step 3:extract features for wiki and moma
step 4:compute cosine similarity
step 5:unify and upgrade artist names that from the same artists
step 6:check if there exists same artists in false_neg_artists list
step 7:upgrade artist map and dataframe based on same_artist_list
step 8:visualize false neg. pairs
step 9:update wiki
step 10:split the second DataFrame into duplicate and unique parts and merge unique part with the first DataFrame

For merging wiki_moma and rijks steps:
step 0:load dataframe for rijks
step 1:extract path combination between first and second dataframe
step 2:extract features for wiki_moma and rijks
step 3:compute cosine similarity
step 4:unify and upgrade artist names that from the same artists
step 5:split the second dataframe into duplicate part and unique part
step 6:check subnames for artists
step 7:upgrade artist names based on subnames
step 8:split rijks into unique part and duplicate part
step 9:filter some genres
step 10:update wikimoma info based on rijks duplicate info
step 11:merge wikimoma with rijks unique part
step 12:add source column