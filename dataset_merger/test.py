import deepdish as dd
import pandas as pd
import copy
import os
import re

# test for checking if duplicate inforation has updated or not
folder_path = '/export/home/jli/workspace/readable_code_data/'
df = pd.read_hdf(os.path.join(folder_path, 'info_wiki_moma_rijks_final_result.hdf5'))
duplicate_wiki_moma = dd.io.load('/export/home/jli/workspace/readable_code_data/split_dup_uniq_info/duplicates_wiki_moma.h5')
duplicate_wm_rijks = dd.io.load('/export/home/jli/workspace/readable_code_data/split_info_wikimoma_rijks/duplicates_wikimoma_rijks_after_substr_detect.h5')
#print duplicate_wiki_moma
print ""
print ""
#print duplicate_wm_rijks
#print df.columns
#with pd.option_context('display.max_rows', None, 'display.max_columns', 50):
#    print df[df['image_id'] == 'rembrandt_st-jerome-kneeling-in-prayer-looking-down-1635']

moma_unique = pd.read_csv(os.path.join(folder_path, 'moma_info_unique.csv'), index_col='id')
print moma_unique.columns
print moma_unique.value_counts()