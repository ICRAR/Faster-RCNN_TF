# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import networks.VGGnet_train
import networks.VGGnet_test
import pdb
import tensorflow as tf

#__sets['VGGnet_train'] = networks.VGGnet_train()

#__sets['VGGnet_test'] = networks.VGGnet_test()

def get_network(name):
    """Get a network by name."""
    nwnm = name.split('_')[1]
    #if not __sets.has_key(name):
    #    raise KeyError('Unknown dataset: {}'.format(name))
    #return __sets[name]
    # if (nwnm in ['train12', 'train13']):
    #     return networks.VGGnet_train(low_level_trainable=True,
    #                                  anchor_scales=[1, 2, 4, 8, 16, 32],
    #                                  anchor_ratios=[1])
    # elif (nwnm in ['test12', 'test13']):
    #     return networks.VGGnet_test(low_level_trainable=True,
    #                                  anchor_scales=[1, 2, 4, 8, 16, 32],
    #                                  anchor_ratios=[1])
    # elif (nwnm in ['train14', 'train15', 'train16', 'train17', 'train18', 'train19', 'train08']):
    #     return networks.VGGnet_train(low_level_trainable=False,
    #                                  anchor_scales=[1, 2, 4, 8, 16, 32],
    #                                  anchor_ratios=[1])
    # elif (nwnm in ['test14', 'test15', 'test16', 'test17', 'test18', 'test19', 'test08']):
    #     return networks.VGGnet_test(low_level_trainable=False,
    #                                  anchor_scales=[1, 2, 4, 8, 16, 32],
    #                                  anchor_ratios=[1])
    # elif (nwnm in ['train07', 'train09', 'train11']):
    #     # we have to re-train low-level ConvNet from scratch
    #     return networks.VGGnet_train(low_level_trainable=True)
    # elif (nwnm in ['test07', 'test09', 'test11']):
    #     # we have to re-train low-level ConvNet from scratch
    #     return networks.VGGnet_test(low_level_trainable=True)
    # elif (nwnm in ['train10']):
    #     return networks.VGGnet_train(low_level_trainable=True,
    #                                  anchor_scales=[2, 4, 8])
    # elif (nwnm in ['test10']):
    #     return networks.VGGnet_test(low_level_trainable=True,
    #                                  anchor_scales=[2, 4, 8])
    if nwnm.find('train') > -1:
        #return networks.VGGnet_test()
        return networks.VGGnet_train(low_level_trainable=False,
                                     anchor_scales=[1, 2, 4, 8, 16, 32],
                                     anchor_ratios=[1], transform_img=True)
    elif nwnm.find('test') > -1:
        #return networks.VGGnet_train()
        return networks.VGGnet_test(low_level_trainable=False,
                                     anchor_scales=[1, 2, 4, 8, 16, 32],
                                     anchor_ratios=[1], transform_img=True)
    else:
        raise KeyError('Unknown dataset: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
