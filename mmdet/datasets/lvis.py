import numpy as np
# from pycocotools.coco import COCO

from .custom import CustomDataset
from .coco import CocoDataset
# from .custom_instaboost import CustomDataset
from .registry import DATASETS
from lvis.lvis import LVIS

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

@DATASETS.register_module
class LvisDataset(CocoDataset):
    # def __init__(self, ann_file, pipeline, data_root=None, img_prefix=None, seg_prefix=None, proposal_file=None, test_mode=False):
    #     super().__init__(ann_file, pipeline, data_root, img_prefix, seg_prefix, proposal_file, test_mode)
    #     self.coco = LVIS(ann_file)
    #     self.img_ids = self.coco.get_img_ids()

    def load_annotations(self, ann_file):
        self.lvis = LVIS(ann_file)
        self.full_cat_ids = self.lvis.get_cat_ids()
        self.full_cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.full_cat_ids)
        }

        self.CLASSES = tuple([item['name'] for item in self.lvis.dataset['categories']])
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        self.img_ids = self.lvis.get_img_ids()
        img_infos = []
        for i in self.img_ids:
            info = self.lvis.load_imgs([i])[0]
            info['filename'] = info['file_name'].split('_')[-1]
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def get_ann_info_withoutparse(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ann_ids)
        return ann_info

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.lvis.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds


    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.img_infos[idx]['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]