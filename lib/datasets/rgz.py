"""
data sets for the RGZ project
"""
import os
import uuid

from datasets.imdb import imdb
from datasets.pascal_voc import pascal_voc
from fast_rcnn.config import cfg

class rgz(pascal_voc):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'rgz_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'RGZ' + self._year)
        self._classes = ('__background__', # always index 0
                         '1_1', '1_2', '1_3', '2_2', '2_3', '3_3')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '_radio.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'PNGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_default_path(self):
        """
        Return the default path where RGZ is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'RGZdevkit' + self._year)
