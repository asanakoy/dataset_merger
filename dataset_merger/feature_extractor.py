from eval.feature_extractor_tf import FeatureExtractorTf
import deepdish as dd
import os

snapshot_path_first = '/export/home/jli/PycharmProjects/practical/data/bvlc_alexnet.tf'
snapshot_path_second = '/export/home/asanakoy/workspace/wikiart/cnn/artist_50/rs_balance/model/snap_iter_335029.tf'
mean_path_second = '/export/home/asanakoy/workspace/wikiart/cnn/artist_50/rs_balance/data/mean.npy'

def feature_extractor(snapshot_path, path_list, data_folder, base_name, mean_path=None):
    """
    extract features
    Args:
        snapshot_path:
        path_list:
        data_folder: folder for saving data
        base_name: name for the saving data
        mean_path:

    Returns: feature matrix

    """
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
    print features.shape
    # save features from the model
    file_name = 'features_' + base_name + '_.h5'
    if not os.path.exists(os.path.join(data_folder, file_name)):
        dd.io.save(os.path.join(data_folder, file_name))
        print 'save features successfully'
    return features