import sys
#sys.path.insert(0, "/usr/lib/python2.7/dist-packages/PILcompat")
#sys.path.insert(0, "/export/home/jli/PycharmProjects/practical/eval/feature_extractor_tf")
from feature_extractor_tf import FeatureExtractorTf
#from feature_extractor_tf import *
#import feature_extractor_tf
import numpy as np
import deepdish as dd
import pickle
import os

snapshot_path_first = '/export/home/jli/PycharmProjects/practical/data/bvlc_alexnet.tf'
snapshot_path_second = '/export/home/asanakoy/workspace/wikiart/cnn/artist_50/rs_balance/model/snap_iter_335029.tf'
mean_path_second = '/export/home/asanakoy/workspace/wikiart/cnn/artist_50/rs_balance/data/mean.npy'
dataset_info = '/export/home/asanakoy/workspace/wikiart/info/info.hdf5'
path_list_pos_1000 = '/export/home/jli/workspace/art_project/practical/data/pos_path_1000'
path_list_neg_1000 = '/export/home/jli/workspace/art_project/practical/data/neg_path_1000'
path_list_ori_1000 = '/export/home/jli/workspace/art_project/practical/data/ori_path_1000'
path_for_save_data = '/export/home/jli/workspace/art_project/practical/data/'
path_list_moma = '/export/home/jli/workspace/art_project/practical/data/pablo-picasso_moma'
path_list_wiki = '/export/home/jli/workspace/art_project/practical/data/pablo-picasso_wiki'
path_list_wiki_all = '/export/home/jli/workspace/art_project/practical/data/path_list_wiki_all'
path_list_moma_all = '/export/home/jli/workspace/art_project/practical/data/path_list_moma_all'

# function: save features to local disk
def save_features(snapshot_path, path_list, path_for_save_data, array_name, mean_path=None):
    net_args = {'gpu_memory_fraction': 0.5,
            'conv5': 'conv5/conv:0', 'fc6':
            'fc6/fc:0', 'fc7': 'fc7/fc:0'}

    if mean_path == None:
        extractor = FeatureExtractorTf(snapshot_path, feature_norm_method=None,
                                               mean_path=None,
                                               net_args=net_args)
    else:
        extractor = FeatureExtractorTf(snapshot_path, feature_norm_method=None,
                                               mean_path=mean_path,
                                               net_args=net_args)

    layers = ['fc7']

    # extract features from image_path
    features = extractor.extract(path_list, layer_names=layers)
    #print type(features)
    print features.shape
    # save features from the model
    #np.save(path_for_save_data + array_name, features)
    #print ""
    #print "features from model"
    #print features.shape
    #print features.shape[0]
    return features


if __name__ == '__main__':

    # save features for original dataset
    if os.path.exists('/export/home/jli/workspace/art_project/practical/data/features_ori_first_1000.npy') is False:
        with open(path_list_ori_1000, 'rb') as fp:
            ori_path_list = pickle.load(fp)

        ori_image_path = []
        array_name_first = 'features_ori_first_1000.npy'
        array_name_second = 'features_ori_second_1000.npy'
        for i in range(len(ori_path_list)):
            im_name = ori_path_list[i] + '.jpg'
            ori_image_path.append(im_name)
        #print ori_image_path
        save_features(snapshot_path_first, ori_image_path, path_for_save_data, array_name_first, mean_path=None)
        save_features(snapshot_path_second, ori_image_path, path_for_save_data, array_name_second, mean_path_second)

    # save features for pos. dataset
    if os.path.exists('/export/home/jli/workspace/art_project/practical/data/features_pos_first_1000.npy') is False:
        with open(path_list_pos_1000, 'rb') as fp:
            pos_path_list = pickle.load(fp)

        array_name_first = 'features_pos_first_1000.npy'
        array_name_second = 'features_pos_second_1000.npy'
        #print ori_image_path
        save_features(snapshot_path_first, pos_path_list, path_for_save_data, array_name_first, mean_path=None)
        save_features(snapshot_path_second, pos_path_list, path_for_save_data, array_name_second, mean_path_second)

    # save features for neg. dataset
    if os.path.exists('/export/home/jli/workspace/art_project/practical/data/features_neg_first_1000.npy') is False:
        with open(path_list_neg_1000, 'rb') as fp:
            neg_path_list = pickle.load(fp)

        neg_image_path = []
        array_name_first = 'features_neg_first_1000.npy'
        array_name_second = 'features_neg_second_1000.npy'
        for i in range(len(neg_path_list)):
            im_name = neg_path_list[i] + '.jpg'
            neg_image_path.append(im_name)
        #print ori_image_path
        save_features(snapshot_path_first, neg_image_path, path_for_save_data, array_name_first, mean_path=None)
        save_features(snapshot_path_second, neg_image_path, path_for_save_data, array_name_second, mean_path_second)

        print "hello..."




    # save features for paintings from picassio in both wikiart and moma
    artist = "picasso"
    moma_fea_first = "fea_moma" + artist + "_first.npy"
    moma_fea_second = "fea_moma" + artist + "_second.npy"
    wiki_fea_first = "fea_wiki" + artist + "_first.npy"
    wiki_fea_second = "fea_wiki" + artist + "_second.npy"
    if os.path.exists(os.path.join(path_for_save_data, moma_fea_first)) is False:
        with open(path_list_moma, 'rb') as fp:
            path_moma = pickle.load(fp)
        with open(path_list_wiki, 'rb') as fp:
            path_wiki = pickle.load(fp)
        save_features(snapshot_path_first, path_wiki, path_for_save_data, wiki_fea_first, mean_path=None)
        save_features(snapshot_path_second, path_wiki, path_for_save_data, wiki_fea_second, mean_path_second)
        save_features(snapshot_path_first, path_moma, path_for_save_data, moma_fea_first, mean_path=None)
        save_features(snapshot_path_second, path_moma, path_for_save_data, moma_fea_second, mean_path_second)

    # save features for all images in wikiart and moma
    moma_fea_first = "fea_all_moma" + "_first.npy"
    moma_fea_second = "fea_all_moma" + "_second.npy"
    wiki_fea_first = "fea_all_wiki" + "_first.npy"
    wiki_fea_second = "fea_all_wiki" + "_second.npy"
    if os.path.exists(os.path.join(path_for_save_data, moma_fea_first)) is False:
        with open(path_list_moma_all, 'rb') as fp:
            path_moma = pickle.load(fp)
        with open(path_list_wiki_all, 'rb') as fp:
            path_wiki = pickle.load(fp)
        save_features(snapshot_path_first, path_wiki, path_for_save_data, wiki_fea_first, mean_path=None)
        save_features(snapshot_path_second, path_wiki, path_for_save_data, wiki_fea_second, mean_path_second)
        save_features(snapshot_path_first, path_moma, path_for_save_data, moma_fea_first, mean_path=None)
        save_features(snapshot_path_second, path_moma, path_for_save_data, moma_fea_second, mean_path_second)

    # combin_wiki_moma_path_list feature extractor
    save_name = 'combine_wiki_moma_features.h5'
    final_folder = '/export/home/jli/workspace/art_project/practical/data/final/'
    if not os.path.exists(os.path.join(final_folder, save_name)):
        wiki_moma_path_list = dd.io.load(os.path.join(final_folder, 'combine_wiki_moma_img_path.h5'))
        features = save_features(snapshot_path_first, wiki_moma_path_list, final_folder, save_name, mean_path=None)
        dd.io.save(os.path.join(final_folder, save_name), features)


    # save features for rijks
    rijks_fea_save_name = "fea_all_rijks.h5"
    if not os.path.exists(os.path.join(final_folder, rijks_fea_save_name)):
        with open(os.path.join(path_for_save_data, 'rijks_img_path_list'), 'rb') as fp:
            path_rijks = pickle.load(fp)
        features = save_features(snapshot_path_first, path_rijks, final_folder, rijks_fea_save_name, mean_path=None)
        dd.io.save(os.path.join(final_folder, rijks_fea_save_name), features)