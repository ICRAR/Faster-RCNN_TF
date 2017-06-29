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
    if (nwnm == 'train07'):
        print('Using networks.VGGnet_train(anchor_scales=[4, 8]')
        return networks.VGGnet_train(anchor_scales=[4, 8])
    elif (nwnm == 'teset07'):
        print('Using networks.VGGnet_test(anchor_scales=[4, 8])')
        return networks.VGGnet_test(anchor_scales=[4, 8])
    elif nwnm.find('test') > -1:
        return networks.VGGnet_test()
    elif nwnm.find('train') > -1:
        return networks.VGGnet_train()
    # elif name.split('_')[1] in ['trainsmall', 'trainfifth', 'trainsixth']:
    #     return networks.VGGnet_train(anchor_scales=[2, 4, 8])
    # elif name.split('_')[1] in ['testsmall', 'testfifth', 'testsixth']:
    #     return networks.VGGnet_test(anchor_scales=[2, 4, 8])
    else:
        raise KeyError('Unknown dataset: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
