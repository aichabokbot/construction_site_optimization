from mrcnn import utils as mrcnn_utils
from tqdm import tqdm
import numpy as np
import json
from utils.utils import get_image_local_path, read_image, get_json_local_path
from utils.utils import convert_annotations


class ChronsiteDataset(mrcnn_utils.Dataset):
    """Derived class of mrcnn Dataset to handle eleven images."""

    def __init__(self, directory_path, sample_ids, verbose=False):
        super().__init__()
        self.DIR_PATH = directory_path
        self.sample_ids = sample_ids
        self.verbose = verbose
        self.source = "Chronsite"
        self.class_names_preset = np.array(['BG',
                                            'Concrete_pump_hose',
                                            'Mixer_truck',
                                            'People',
                                            'Vertical_formwork'])

    def _get_id_of_class(self, clabel):
        if clabel not in self.class_names_preset:
            raise Exception(f'Class {clabel} not present in dataset classes: {self.class_names_preset}')
        return np.where(self.class_names_preset == clabel)[0][0]

    def load_image(self, image_id, zero_image=False):
        """ Load the images during training. """
        info = self.image_info[image_id]
        image_id = info['id']
        image_path = get_image_local_path(image_id, self.DIR_PATH)
        image = read_image(image_path)
        return image

    def load_mask(self, image_id):
        """ Load the masks during training. """
        info = self.image_info[image_id]
        image_id = info['id'] 
        mask_path = get_json_local_path(image_id, self.DIR_PATH)
        with open(mask_path) as f:
            annotations = json.load(f)
        masks, class_names = convert_annotations(annotations, info['image_size'][:-1])
        class_ids = np.array([self._get_id_of_class(c) for c in class_names], dtype=np.int32)
        return masks, class_ids

    def load_kernel(self) -> None:
        """ Load the metadate before training. """
        self.classes = set()
        for sample_id in tqdm(self.sample_ids):
            try:
                image_path = get_image_local_path(sample_id, self.DIR_PATH)
                image = read_image(image_path)
                annotation_path = get_json_local_path(sample_id, self.DIR_PATH)
                with open(annotation_path) as f:
                    annotations = json.load(f)
                if (len(annotations["objects"]) == 0):
                    print(f'Skip sample {sample_id}, no gt_objects')
                    continue
                for obj in annotations["objects"]:
                    self.classes.update({obj["classTitle"]})
                self.add_image(source=self.source,
                               image_id=sample_id,
                               sub_class_label="ConstructionSite",
                               path=image_path,
                               image_size=image.shape,
                            )
            except Exception as e:
                print(f'EXCP: {e} during handling {sample_id}')
                continue
        for i, c in enumerate(list(self.class_names_preset[1:])):
            self.add_class(self.source, i + 1, c)
        self.prepare()
        return


class ChronsiteTestDataset(ChronsiteDataset):
    """Derived class of mrcnn Dataset to handle test set images."""

    def __init__(self, directory_path, sample_ids, verbose=False):
        super().__init__(directory_path, sample_ids, verbose)

    def load_mask(self, image_id):
        raise NotImplementedError("No ground truth annotations for samples in the test set.")

    def load_kernel(self) -> None:
        """ Load the dataset before inference. """
        self.classes = set()
        for sample_id in tqdm(self.sample_ids):
            try:
                image_path = get_image_local_path(sample_id, self.DIR_PATH)
                image = read_image(image_path)

                self.add_image(
                    source=self.source,
                    image_id=sample_id,
                    sub_class_label="ConstructionSite",
                    path=image_path,
                    image_size=image.shape,
                )
            except Exception as e:
                print(f'EXCP: {e} during handling {sample_id}')
                continue
        for i, c in enumerate(list(self.class_names_preset[1:])):
            self.add_class(self.source, i + 1, c)
        self.prepare()
        return
