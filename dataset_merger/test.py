import deepdish as dd
import pandas as pd
import copy
import time
import os
import re

# test for checking if duplicate inforation has updated or not
folder_path = '/export/home/jli/workspace/data_after_run/'
df = pd.read_hdf(os.path.join(folder_path, 'info_wiki_moma_rijks_final_result.hdf5'))
duplicate_wiki_moma = dd.io.load('/export/home/jli/workspace/data_after_run/duplicates_wiki_moma.h5')
duplicate_wm_rijks = dd.io.load('/export/home/jli/workspace/readable_code_data/split_info_wikimoma_rijks/duplicates_wikimoma_rijks_after_substr_detect.h5')
#print duplicate_wm_rijks
#print df[df.image_id == 'laszlo-moholy-nagy_yellow-circle']
#print duplicate_wm_rijks
#print df.columns


print df[df.image_id == 'jackson-pollock_mask'].is_duplicate