
# coding: utf-8

# In[22]:


"""
Check all datasets
"""
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'load_ext autotime')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-deep')
from ipywidgets import interact
from ipywidgets import Controller

import numpy as np
import os
import sys
from tqdm import tqdm
import pandas as pd
sys.path.append('/export/home/asanakoy/workspace/neural_network')
sys.path.append('/export/home/asanakoy/workspace/art_datasets')
sys.path.append('/export/home/asanakoy/workspace/dataset_merger_old')
import make_data.dataset
import wikiart.info.preprocess_info
import dataset_merger.read_datasets


# In[4]:


dfs = dataset_merger.read_datasets.read_datasets()


# In[6]:


dfs['googleart']


# In[16]:



artists_df = pd.read_hdf('/export/home/asanakoy/workspace/googleart/info/all_artists.hdf5')
print pd.notnull(artists_df['wiki_url']).sum()
artists_df.index = artists_df.artist_id
assert not artists_df.index.has_duplicates
artists_df['artist_slug'] = artists_df['name'].str.lower().str.strip().str.replace(' ', '-')
assert np.all(artists_df.loc[dfs['googleart']['artist_id'], 'artist_slug'].values == dfs['googleart']['artist_slug'].values)
artists_df


# In[26]:


dfs['googleart']['artist_wiki_url'] = artists_df.loc[dfs['googleart']['artist_id'], 'wiki_url'].values
dfs['googleart']['artist_id_wiki'] = dfs['googleart']['artist_wiki_url'].apply(lambda x: os.path.basename(x) if isinstance(x, unicode) else x)
dfs['googleart']


# In[27]:


# TODO: merge artist according to wiki urls (mb degruyter as well)


artists_df.wiki_url.value_counts()


# In[17]:


total_artists_num = 0
artist_names = list()
for key, df in dfs.iteritems():
    cur_artist_names = df.artist_name.unique().tolist()
    artist_names.extend(cur_artist_names)
    cur_artists_num = len(df.artist_name.unique())
    print key, 'unique artists:', cur_artists_num
    total_artists_num += cur_artists_num
print '--'
print 'Total unique artists num:', total_artists_num 
artist_names = np.unique(artist_names)
print len(artist_names)


# In[ ]:





# In[19]:


artist_names = sorted(artist_names)
for x in artist_names:
    print x


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




