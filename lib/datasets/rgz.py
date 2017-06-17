"""
data sets for the RGZ project
"""

from datasets.pascal_voc import pascal_voc

class rgz_imdb(pascal_voc):
    def __init__(self, image_set, year, devkit_path=None):
        super(rgz_imdb, self).__init__(image_set, year, devkit_path)
        self._classes = ('__background__', # always index 0
                         'one', 'two', 'three', 'four', 'five', 'six')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'

    def _get_default_path(self):
        """
        Return the default path where RGZ is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'RGZdevkit' + self._year)
