from credentials.blob_credentials import facts_sas_token, facts_container
from azure.storage.blob import ContainerClient, BlobClient
import pandas as pd
import os
import json
import colorsys
import random
import numpy as np
import cv2
import imageio
from matplotlib import patches
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from datetime import datetime
import zlib
import base64
from mrcnn.model import MaskRCNN
from mrcnn.utils import resize_image

account_url = "https://hecdf.blob.core.windows.net"
facts_blob_service = ContainerClient(account_url=account_url,
                                     container_name=facts_container,
                                     credential=facts_sas_token)


# ===============================
# INLINES
# ===============================


def get_people_local_path(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, sample_id + '_people.json')


def get_poly_local_path(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, sample_id + '_poly.json')


def get_image_local_path(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, sample_id + '.jpg')


def get_result_local_path(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, sample_id + '.npz')


def get_json_local_path(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, sample_id + '.json')


def get_inference_result_path(sample_id: str, directory_path: str) -> str:
    return os.path.join(directory_path, sample_id + '.json')


# ===============================
# INFERENCE MODE
# ===============================


def filter_resuLt_by_score(result: dict, thres: float = 0.9) -> dict:
    """Filter the inference results by the given confidence."""
    filters = (result['scores'] >= thres)
    fitered_result = {}
    fitered_result['rois'] = result['rois'][filters]
    fitered_result['masks'] = []
    for i in range(len(filters)):
        if filters[i]:
            fitered_result['masks'].append(result['masks'][..., i])
    fitered_result['masks'] = np.stack(fitered_result['masks'], axis=-1)
    fitered_result['class_ids'] = result['class_ids'][filters]
    fitered_result['scores'] = result['scores'][filters]
    return fitered_result


def retrieve_inference_to_json(model: MaskRCNN, ds,
                               image_id: str, json_dir: str) -> None:
    """Retrieve inference result from the model the store in json file."""
    # Load image from dataset
    original_image = ds.load_image(image_id)
    _, window, scale, padding, _ = resize_image(original_image, mode="pad64")
    # No rescaling applied
    assert scale == 1

    # Retrieve predictions
    height, width = original_image.shape[:2]
    result = model.detect([original_image], verbose=0)[0]
    filtered_result = filter_resuLt_by_score(result, 0.9)

    # Dict object to dump to json
    dump = {}
    dump["image"] = {
        "file_name": 'img/' + ds.image_info[image_id]['id'] + '.jpg',
        "id": int(image_id),
        "height": int(height),
        "width": int(width),
    }

    dump["annotations"] = []

    assert filtered_result['rois'].shape[0] == \
           filtered_result['masks'].shape[-1] == \
           filtered_result['class_ids'].shape[0]

    # Encoding annotations into json
    for obj_id in range(filtered_result['rois'].shape[0]):
        roi = filtered_result['rois'][obj_id, :]
        mask = filtered_result['masks'][..., obj_id]
        class_id = filtered_result['class_ids'][obj_id]
        y1, x1, y2, x2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        polygon = []

        # 1d flatten list of [x, y] coordinates
        for pt_id in range(cnt.shape[0]):
            polygon.append(int(cnt[pt_id, 0, 0]))
            polygon.append(int(cnt[pt_id, 0, 1]))

        obj = {'id': int(obj_id),
               'segmentation': [polygon],
               'area': float(cv2.contourArea(cnt)),
               # x, y, h, w
               'bbox': [x1, y1, x2 - x1, y2 - y1],
               'image_id': int(image_id),
               'category_id': int(class_id),
               'iscrowd': 0}
        dump["annotations"].append(obj)

    json_path = get_inference_result_path(ds.image_info[image_id]['id'],
                                          json_dir)

    with open(json_path, 'w') as f:
        json.dump(dump, f)
    return


def visualize_inference_sample(sample_id: str,
                               class_names: list,
                               image_dir: str,
                               json_dir: str,
                               render_dir: str) -> None:
    """Load inference result json file and render overlay on raw image."""
    image_path = get_image_local_path(sample_id, image_dir)
    json_path = get_inference_result_path(sample_id, json_dir)
    render_path = os.path.join(render_dir, sample_id + '.jpg')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(json_path, 'r') as f:
        data = json.load(f)

    assert image.shape[0] == data["image"]["height"]
    assert image.shape[1] == data["image"]["width"]

    fig, ax = plt.subplots(1, figsize=(20,20))
    colors = random_colors(len(data["annotations"]))
    height, width = image.shape[:2]

    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(f"Sample_id: {sample_id}, shape {image.shape}")

    masked_image = image.astype(np.uint32).copy()

    for i, instance in enumerate(data["annotations"]):
        x, y, w, h = instance["bbox"]
        p = patches.Rectangle((x, y), w, h, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=colors[i], facecolor='none')
        ax.add_patch(p)

        polygon = instance["segmentation"][0]
        polygon = np.array(polygon).reshape(-1, 2)

        p = Polygon(polygon, facecolor="none", edgecolor=colors[i],
                    linewidth=None, fill=True)
        p.set_fill(True)
        ax.add_patch(p)

        # Label
        ax.text(x, y + 8, class_names[instance["category_id"]],
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(render_path)
    plt.close(fig)
    return


def dump_coco_format(sample_ids: list,
                     json_dir: str,
                     coco_json_path: str,
                     class_names: list) -> None:
    """Merge individual json files into an integrated one."""
    dump_dict = {
        "categories": [],
        "info": {
            "description": "Chronsite test set, Group 8.",
            "year": 2020
        },
        "licenses": [],
        "images": [],
        "annotations": []
    }

    for class_id, class_name in enumerate(class_names):
        dump_dict["categories"].append({
            "Supercategory": "none",
            "name": class_name,
            "id": class_id
        })

    annotation_counter = 0
    for image_id, sample_id in enumerate(sample_ids):
        json_path = get_inference_result_path(sample_id, json_dir)

        with open(json_path, 'r') as fp:
            data = json.load(fp)
        image_meta = data["image"]
        image_meta["id"] = image_id
        dump_dict["images"].append(image_meta)

        for anno in data["annotations"]:
            annotation_counter += 1
            anno["image_id"] = image_id
            anno["id"] = annotation_counter
            dump_dict["annotations"].append(anno)

    with open(coco_json_path, 'w') as fp:
        json.dump(dump_dict, fp)
    return


def verify_coco_json(json_path: str) -> None:
    """Check integraty of the big json file in coco format. """
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    print(f"CALSSES {len(data['categories'])} classes:")
    class_id_name = {}
    for cat in data["categories"]:
        class_id_name.update({cat['id']: cat['name']})
    print(class_id_name)
    print(" ")
    print(f'INFO: {data["info"]}')
    print(" ")
    print(f'LICENSES: {data["licenses"]}')
    print(" ")
    print("IMAGES")
    shapes = set()
    image_names = []
    image_id = set()
    for image in data["images"]:
        shapes.add((image["height"], image["width"]))
        image_names.append(image["file_name"])
        image_id.add(image["id"])
        assert len(image_names) == len(image_id)
    print(f"Shapes: {shapes}")
    print(f"Filenames: {image_names}")
    print(f"ID: {image_id}")
    print(" ")
    print("SEGMENTATIONS")
    anno_id = set()
    for obj in data["annotations"]:
        anno_id.add(obj["id"])
        assert obj["image_id"] in image_id
        assert obj["category_id"] in list(class_id_name.keys())

    assert len(anno_id) == len(data["annotations"])
    print(f"number of annotations {len(data['annotations'])}")
    return


# ===============================
# EXPOLORE DATA
# ===============================

def get_sample_role(sample_id: str,
                    train_sample_list: list,
                    val_sample_list: list) -> str:
    """Query the sample is in which bucket."""
    if sample_id in train_sample_list:
        return "train"
    elif sample_id in val_sample_list:
        return "val"
    return None


def get_facts_blobs() -> list:
    """Query all blobs from azure storage."""
    blobs = list(facts_blob_service.list_blobs())
    return blobs


def fetch_train_set() -> pd.DataFrame:
    """Query all samples from azure storage."""
    blobs = list(facts_blob_service.list_blobs())
    records = {}

    for blob in blobs:
        # remove irrelevant files
        file_name = blob.name
        if ".DS_Store" in file_name:
            continue

        split_file_name = file_name.split('/')

        # Detection_Train_Set_bis
        if "_Bis" in split_file_name[0]:
            pattern = ".jpg"
            index = split_file_name[2].find(pattern)
            date_time_string = split_file_name[2][:index]
            sample_id = date_time_string
            try:
                date_time = datetime.strptime(date_time_string, "%Y_%m_%d_%H_%M_%S")
            except Exception:
                date_time = None
            if ".json" in file_name:
                # Annotation
                record = {
                    "sample_id": date_time_string,
                    "construction_site": "Analytic",
                    "annotation_blob_id": file_name,
                    "date_time": date_time,
                }
            else:
                # Image
                record = {
                    "sample_id": date_time_string,
                    "construction_site": "Analytic",
                    "image_blob_id": file_name,
                    "date_time": date_time,
                }
        elif split_file_name[0] == "Analytics_Train_Set":
            pattern = ".jpg"
            index = split_file_name[-1].find(pattern)
            date_time_string = split_file_name[-1][:index]
            sample_id = date_time_string

            try:
                date_time = datetime.strptime(date_time_string, "%Y-%m-%d-%H-%M-%S")
            except Exception:
                date_time = None

            if ".json" in file_name:
                # Annotation
                if "people" in file_name:
                    record = {
                        "sample_id": sample_id,
                        "construction_site": "Analytic2",
                        "annotation_people_blob_id": file_name,
                        "date_time": date_time,
                    }

                elif "poly" in file_name:
                    record = {
                        "sample_id": sample_id,
                        "construction_site": "Analytic2",
                        "annotation_poly_blob_id": file_name,
                        "date_time": date_time,
                    }

            else:
                record = {
                    "sample_id": sample_id,
                    "construction_site": "Analytic2",
                    "image_blob_id": file_name,
                    "date_time": date_time,
                }

        else:
            # Detection_Train_Set
            pattern1 = "Batch2__"
            pattern2 = "frame"
            pattern3 = ".jpg"

            index1 = split_file_name[2].find(pattern1)
            index2 = split_file_name[2].find(pattern2)
            index3 = split_file_name[2].find(pattern3)

            construction_site_name = split_file_name[2][index1+len(pattern1):index2]
            sample_id = split_file_name[2][:index3]

            if ".json" in file_name:
                # Annotation
                record = {
                    "sample_id": sample_id,
                    "construction_site": construction_site_name,
                    "annotation_blob_id": file_name,
                    "date_time": None,
                }
            else:
                # Image
                record = {
                    "sample_id": sample_id,
                    "construction_site": construction_site_name,
                    "image_blob_id": file_name,
                    "date_time": None,
                }

        if sample_id in records:
            records[sample_id].update(record)
        else:
            records[sample_id] = record

    records = list(records.values())
    records = [l for l in records if (l.get("image_blob_id")) and
               ((l.get("annotation_blob_id")) or ((l.get("annotation_poly_blob_id"))
                and (l.get("annotation_people_blob_id"))))]
    return pd.DataFrame(records)


def download_blob(blob_id: str, local_path: str) -> None:
    """Download file from remote storage to local path."""
    bc = BlobClient(account_url=account_url, container_name=facts_container,
                    blob_name=blob_id, snapshot=None,
                    credential=facts_sas_token)
    with open(local_path, "wb") as download_file:
        download_file.write(bc.download_blob().readall())
    return


def download_sample(image_blob: str, annotation_blob: str,
                    sample_id: str, directory_path: str,
                    force_refresh: bool = False) -> None:
    """Download image and json for a given sample id"""
    image_local_path = get_image_local_path(sample_id, directory_path)
    json_local_path = get_json_local_path(sample_id, directory_path)
    if ((force_refresh) or (not os.path.exists(image_local_path))):
        download_blob(image_blob, image_local_path)
    if ((force_refresh) or (not os.path.exists(json_local_path))):
        download_blob(annotation_blob, json_local_path)
    return


def download_sample_analytics(image_blob: str,
                              annotation_people_blob: str,
                              annotation_poly_blob: str,
                              sample_id: str,
                              directory_path: str,
                              force_refresh: bool = False) -> None:
    """Download image, people_json and poly_json for a given sample id"""
    image_local_path = get_image_local_path(sample_id, directory_path)
    people_local_path = get_people_local_path(sample_id, directory_path)
    poly_local_path = get_poly_local_path(sample_id, directory_path)
    if ((force_refresh) or (not os.path.exists(image_local_path))):
        download_blob(image_blob, image_local_path)
    if ((force_refresh) or (not os.path.exists(people_local_path))):
        download_blob(annotation_people_blob, people_local_path)
    if ((force_refresh) or (not os.path.exists(poly_local_path))):
        download_blob(annotation_poly_blob, poly_local_path)
    return


def visualize_sample(sample_id: str, directory_path: str) -> None:
    """Render the image and annotations for a given sample id. """
    image = cv2.imread(get_image_local_path(sample_id, directory_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(get_json_local_path(sample_id, directory_path)) as f:
        data = json.load(f)
    _, ax = plt.subplots(1, figsize=(20, 20))
    colors = random_colors(len(data["objects"]))
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(f"Sample_id: {sample_id}, shape {image.shape}")

    masked_image = image.astype(np.uint32).copy()

    for i, instance in enumerate(data["objects"]):
        if instance["geometryType"] == "rectangle":
            y1, x1, y2, x2 = instance["points"]["exterior"][0][1], \
                             instance["points"]["exterior"][0][0], \
                             instance["points"]["exterior"][1][1], \
                             instance["points"]["exterior"][1][0]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=colors[i], facecolor='none')
            ax.add_patch(p)

        if instance["geometryType"] == "polygon":
            y1, x1 = instance["points"]["exterior"][0][1], \
                     instance["points"]["exterior"][0][0]
            p = Polygon(instance["points"]["exterior"],
                        facecolor="none", edgecolor=colors[i],
                        linewidth=None, fill=True)
            p.set_fill(True)
            ax.add_patch(p)

        # Label
        ax.text(x1, y1 + 8, instance["classTitle"],
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()
    return


def visualize_sample_analytic(sample_id: str, directory_path: str) -> None:
    """Render the image and annotations for a given sample id in analytic set."""
    image = cv2.imread(get_image_local_path(sample_id, directory_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(get_people_local_path(sample_id, directory_path)) as f:
        annotation_people = json.load(f)
    with open(get_poly_local_path(sample_id, directory_path)) as f:
        annotation_poly = json.load(f)
    _, ax = plt.subplots(1, figsize=(20, 20))
    print(f"{len(annotation_people['objects'])} peoples + {len(annotation_poly['objects'])} polygons")
    colors = random_colors(len(annotation_people["objects"]) +
                           len(annotation_poly["objects"]))
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(f"Sample_id: {sample_id}, shape {image.shape}")

    masked_image = image.astype(np.uint32).copy()

    for i, instance in enumerate(annotation_people["objects"]):
        if instance["geometryType"] == "rectangle":
            y1, x1, y2, x2 = instance["points"]["exterior"][0][1], \
                             instance["points"]["exterior"][0][0], \
                             instance["points"]["exterior"][1][1], \
                             instance["points"]["exterior"][1][0]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=colors[i], facecolor='none')
            ax.add_patch(p)

        # Label
        ax.text(x1, y1 + 8, instance["classTitle"],
                color='w', size=11, backgroundcolor="none")

    for j, instance in enumerate(annotation_poly["objects"]):
        if instance["geometryType"] == "bitmap":
            mask = base64_2_mask(instance["bitmap"]["data"])
            x1, y1 = instance["bitmap"]["origin"][0], \
                instance["bitmap"]["origin"][1]
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            cnt = np.squeeze(contours[0], axis=1)

            # Depends on whether the mini-mask was adpoted
            # cnt += [x1, y1]

            p = Polygon(cnt, facecolor="none",
                        edgecolor=colors[len(annotation_people["objects"])+j],
                        linewidth=None, fill=True)
            p.set_fill(True)
            ax.add_patch(p)

        # Label
        ax.text(x1, y1 + 8, instance["classTitle"],
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()
    return


def render_heatmap(df: pd.DataFrame, local_directory: str) -> None:
    """Visualize the heatmap of concentration of workers."""
    image = read_image(get_image_local_path(df.iloc[-1, 0],  local_directory))
    heatmap = np.zeros(image.shape[:2], dtype=np.float)
    for index, row in df.iterrows():
        sample_id = row["sample_id"]
        mask_path = get_people_local_path(sample_id, local_directory)
        with open(mask_path) as f:
            annotations = json.load(f)
        for obj in annotations["objects"]:
            if obj["classTitle"] == "People_model":
                mask = np.zeros((1024, 1280), dtype=np.float)
                mask = draw_rectangle(mask, obj['points']['exterior'])
                heatmap += mask
    kernel = np.ones((10, 10), np.float32)/25
    heatmap = cv2.filter2D(heatmap, -1, kernel)

    heatmap_render = (heatmap / np.max(np.max(heatmap))) * 255.0
    heatmap_render = cv2.applyColorMap(heatmap_render.astype(np.uint8),
                                       cv2.COLORMAP_RAINBOW)
    heatmap_mask = heatmap >= np.max(np.max(heatmap))*0.01

    image[heatmap_mask] = image[heatmap_mask] * 0.5 + \
        heatmap_render[heatmap_mask] * 0.5
    _ = plt.figure(figsize=(12, 12))
    plt.axis('off')
    print(image.shape)
    plt.imshow(image)
    return


# ===============================
# EVALUATION FUNCTIONS
# ===============================


def get_precision_recall(df: pd.DataFrame) -> (list, list):
    """Calculate P/R pairs for a range of confidence_levels"""
    precisions = [0]
    recalls = [1]
    for confidence_level in np.arange(0, 1.0, 0.01):
        tp = len(df[(df.pred_class == df.gt_class) &
                 (df.pred_score >= confidence_level)])
        fn = len(df[((df.pred_class < 0) |
                 (df.pred_score < confidence_level)) &
                 (df.gt_class > -1)])
        fp = len(df[((df.pred_class > -1) &
                 (df.pred_score >= confidence_level)) &
                 (df.gt_class < 0)])
        precisions.append(tp/(tp+fp+1e-5))
        recalls.append(tp/(tp+fn+1e-5))
    precisions.append(1)
    recalls.append(0)
    return precisions, recalls


def calculate_ap_from_pr(precisions: list, recalls: list) -> float:
    """Calculate Average Percision from P-R pairs"""
    AP = 0
    for i in range(len(precisions) - 1):
        AP += (recalls[i] - recalls[i+1]) * (precisions[i] + precisions[i+1]) / 2
    return AP


def get_object_matches(result: dict,
                       pred_matches: list,
                       gt_matches: list,
                       gt_class_id: list,
                       image_id: int,
                       IoUs: list) -> list:
    """Sort and merge the matched objects between gt and pred."""
    IoUs = np.max(IoUs, axis=1)
    assert(len(IoUs) == len(pred_matches))
    object_matches = []
    for pred_score, pred_match, pred_class_id, IoU in zip(result["scores"], pred_matches, result["class_ids"], IoUs):
        pred_class = pred_class_id
        gt_class = gt_class_id[int(pred_match)] if pred_match > -1 else -1

        object_matches.append({"image_id": image_id,
                               "pred_class": int(pred_class),
                               "gt_class": int(gt_class),
                               "pred_score": pred_score,
                               "highest_mIoU": IoU})

    for gt_match, gt_class in zip(gt_matches, gt_class_id):
        if gt_match == -1:
            object_matches.append({"image_id": image_id,
                                   "pred_class": -1,
                                   "gt_class": int(gt_class),
                                   "pred_score": 0,
                                   "highest_mIoU": 0})
    return object_matches


# ===============================
# MISCELLANEOUS FUNCTIONS
# ===============================


def read_imageio(image_path: str):
    """Load image by imageio library."""
    image = imageio.imread(image_path)
    return image


def read_image(image_path: str):
    """Load image by opencv library."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def random_colors(N: int, bright=True) -> list:
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def draw_rectangle(mask, bbox):
    """Helper function for rendering."""
    y1, x1, y2, x2 = bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]
    contour = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    contour = np.array(contour, dtype=np.int32)
    cv2.drawContours(mask, [contour], -1, (255), -1)
    return mask


def draw_polygon(mask, contour):
    """Helper function for rendering."""
    contour = np.array(contour, dtype=np.int32)
    cv2.drawContours(mask, [contour], -1, (255), -1)
    return mask


def convert_annotations(annotations, image_shape):
    """Convert masks from polygon to binary mask for training."""
    masks = []
    class_names = []
    for obj in annotations["objects"]:
        class_name = obj["classTitle"]
        mask = np.zeros(image_shape, dtype=np.int8)
        if (obj["geometryType"] == "rectangle"):
            mask = draw_rectangle(mask, obj['points']['exterior'])
        elif (obj["geometryType"] == "polygon"):
            mask = draw_polygon(mask, obj['points']['exterior'])
        masks.append(mask)
        class_names.append(class_name)
    masks = np.stack(masks, axis=2)
    return masks, class_names


def split_train_val_dataframe(df_all, split_frac=0.2, verbose=False):
    """Split train valid dataset."""
    construction_site_meta = list(df_all.construction_site.value_counts().items())
    train_sample_ids = []
    val_sample_ids = []
    for construction_site, num_tot in construction_site_meta:
        # print(construction_site, num_tot)
        sample_ids = list(df_all[df_all.construction_site == construction_site].sample_id.values)
        assert(len(sample_ids) == num_tot)
        split_index = int(num_tot * split_frac)
        random.shuffle(sample_ids)
        val_sample_ids += sample_ids[:split_index]
        train_sample_ids += sample_ids[split_index:]
        if verbose:
            print(f'Construction site {construction_site}, total samples {num_tot}')
            print(f'train samples {len(sample_ids[split_index:])} , valid samples {len(sample_ids[:split_index])}')
    return train_sample_ids, val_sample_ids


def base64_2_mask(s):
    """Convert bitmap string to binary mask."""
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask
