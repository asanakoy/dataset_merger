import os
import numpy as np
import sklearn.preprocessing
import tensorflow as tf

#from tfproj
import eval.features
import eval.image_getter
import sys
print sys.path
#debug
import tfext
from tqdm import tqdm
import math


class FeatureExtractorTf(object):
    def __init__(self, snapshot_path,
                 feature_norm_method=None, mean_path=None,
                 net_args=None,
                 img_resize_shape=(227, 227)):
        if not os.path.exists(snapshot_path):
            raise IOError('File not found: {}'.format(snapshot_path))

        if not isinstance(feature_norm_method, list):
            feature_norm_method = [feature_norm_method]
        accepable_methods = [None, 'signed_sqrt', 'unit_norm']
        for method in feature_norm_method:
            if method not in accepable_methods:
                raise ValueError('unknown norm method: {}. Use one of {}'.format(method, accepable_methods))
        if img_resize_shape is not None and len(img_resize_shape) != 2:
            raise ValueError('img_resize_shape must be None or (h, w)')

        self.img_resize_shape = img_resize_shape
        self.feature_norm_method = feature_norm_method

        self.mean = None
        if mean_path is not None:
            assert mean_path.endswith('.npy')
            self.mean = np.load(mean_path)
        self.net = eval.features.load_net_with_graph(snapshot_path, **net_args)
        self.rgb_batch = False

    def extract_one(self, image_path, layer_name, flipped=False):
        """
        Extract features from the image
        """
        if not isinstance(image_path, basestring) and not isinstance(image_path, unicode):
            raise TypeError('image_path must be a string!')
        if len(self.feature_norm_method) > 1:
            raise NotImplementedError()
        image_getter = eval.image_getter.ImageGetterFromPaths([image_path],
                                                              im_shape=self.img_resize_shape,
                                                              rgb_batch=self.rgb_batch)

        feature_dict = eval.features.extract_features(flipped, self.net,
                                                      layer_names=[layer_name],
                                                      image_getter=image_getter,
                                                      mean=self.mean,
                                                      verbose=False)

        # feed to the net_stream augmented images anf pool features after
        features = feature_dict[layer_name]
        assert len(features) == 1
        features = features[0]
        if 'unit_norm' in self.feature_norm_method:
            features /= np.linalg.norm(features)
        return features

    def extract(self, image_paths, layer_names, flipped=False, verbose=2):
        """
        Extract features from the image
        """
        try:
            image_paths.__getattribute__('__len__')
        except AttributeError:
            raise TypeError('image_paths must be a container of paths')
        if len(self.feature_norm_method) > 1:
            raise NotImplementedError()
        if not isinstance(layer_names, list):
            layer_names = [layer_names]

        image_getter = eval.image_getter.ImageGetterFromPaths(image_paths,
                                                              im_shape=self.img_resize_shape,
                                                              rgb_batch=self.rgb_batch)

        feature_dict = eval.features.extract_features(flipped, self.net,
                                                      layer_names=layer_names,
                                                      image_getter=image_getter,
                                                      mean=self.mean,
                                                      verbose=verbose,
                                                      should_reshape_vectors=True)

        # feed to the net_stream augmented images anf pool features after
        features = np.hstack(feature_dict.values())
        if 'unit_norm' in self.feature_norm_method:
            sklearn.preprocessing.normalize(features, norm='l2', axis=1, copy=False)
        assert len(features) == len(image_paths)
        return features


class BilinearFeatureExtractorTf(FeatureExtractorTf):
    def extract(self, image_paths, layer_names, flipped=False, verbose=2):
        try:
            image_paths.__getattribute__('__len__')
        except AttributeError:
            raise TypeError('image_paths must be a container of paths')
        if not isinstance(layer_names, list):
            layer_names = [layer_names]
        if len(layer_names) > 1:
            raise NotImplementedError

        image_getter = eval.image_getter.ImageGetterFromPaths(image_paths,
                                                              im_shape=self.img_resize_shape,
                                                              rgb_batch=self.rgb_batch)

        feature_dict = eval.features.extract_features(flipped, self.net,
                                                      layer_names=layer_names,
                                                      image_getter=image_getter,
                                                      mean=self.mean,
                                                      verbose=verbose,
                                                      should_reshape_vectors=False)

        # feed to the net_stream augmented images anf pool features after
        features = feature_dict[layer_names[0]]

        feature_map_size = int(np.prod(features.shape[1:-1]))
        features = features.reshape((features.shape[0], feature_map_size, features.shape[-1]))

        bilinear_features = np.zeros((features.shape[0], features.shape[-1] ** 2),
                                     dtype=np.double)
        for i in xrange(bilinear_features.shape[0]):
            bilinear_features[i, ...] = np.dot(features[i, ...].T, features[i, ...]).reshape(
                -1)

        if 'signed_sqrt' in self.feature_norm_method:
            signs = np.sign(bilinear_features)
            bilinear_features = signs * np.sqrt(signs * bilinear_features)
        if 'unit_norm' in self.feature_norm_method:
            sklearn.preprocessing.normalize(bilinear_features, norm='l2', axis=1, copy=False)

        print 'bilinear_features.shape=', bilinear_features.shape
        return bilinear_features