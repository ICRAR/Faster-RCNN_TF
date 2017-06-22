"""
data sets for the RGZ project
"""
import os

from datasets.pascal_voc import pascal_voc
from fast_rcnn.config import cfg

class rgz(pascal_voc):
    def __init__(self, image_set, year, devkit_path=None):
        super(rgz_imdb, self).__init__(image_set, year, devkit_path)
        # self._classes = ('__background__', # always index 0
        #                  '1C_1P', '1C_2P', '1C_3P', '1C_4P', '1C_>4P',
        #                  '2C_2P', '2C_3P', '2C_3P', '2C_4P', '2C_>4P',
        #                  '3C_3P', '3C_4P', '3C_>4P', '4C_>=4P', '>=5C_>=5P')
        self._classes = ('__background__', # always index 0
                         '1_1', '1_2', '1_3', '2_2', '2_3', '3_3')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._data_path = os.path.join(self._devkit_path, 'RGZ' + self._year)

    def _get_default_path(self):
        """
        Return the default path where RGZ is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'RGZdevkit' + self._year)
