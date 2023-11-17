"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
import sys
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import itertools
import numpy as np
import PIL
from PIL import Image
import math
import torch
from torchvision import transforms

# Custom scripts to import
sys.path.append("detr")
import util.misc as utils
import datasets.transforms as R
import transforms as T


def read_pascal_voc(xml_file: str, class_map=None):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []

    for object_ in root.iter("object"):

        filename = root.find("filename").text

        ymin, xmin, ymax, xmax = None, None, None, None

        label = object_.find("name").text
        try:
            label = int(label)
        except:
            label = int(class_map[label])

        for box in object_.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        bbox = [xmin, ymin, xmax, ymax]  # PASCAL VOC

        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels


def crop_around_bbox_coco(image, crop_bbox, max_margin, target):
    width, height = image.size
    left = max(1, int(round(crop_bbox[0] - max_margin * random.random())))
    top = max(1, int(round(crop_bbox[1] - max_margin * random.random())))
    right = min(width, int(round(crop_bbox[2] + max_margin * random.random())))
    bottom = min(height, int(round(crop_bbox[3] + max_margin * random.random())))
    cropped_image = image.crop((left, top, right, bottom))
    cropped_bboxes = []
    cropped_labels = []
    for bbox, label in zip(target["boxes"], target["labels"]):
        bbox = list_bbox_cxcywh_to_xyxy(bbox)
        bbox = [
            max(bbox[0], left) - left,
            max(bbox[1], top) - top,
            min(bbox[2], right) - left,
            min(bbox[3], bottom) - top,
        ]
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            bbox = list_bbox_xyxy_to_cxcywh(bbox)
            cropped_bboxes.append(bbox)
            cropped_labels.append(label)

    if len(cropped_bboxes) > 0:
        target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(cropped_labels, dtype=torch.int64)
        w, h = img.size
        target["size"] = torch.tensor([w, h])
        return cropped_image, target

    return image, target


class TightAnnotationCrop(object):
    def __init__(
        self, labels, left_max_pad, top_max_pad, right_max_pad, bottom_max_pad
    ):
        self.labels = set(labels)
        self.left_max_pad = left_max_pad
        self.top_max_pad = top_max_pad
        self.right_max_pad = right_max_pad
        self.bottom_max_pad = bottom_max_pad

    def __call__(self, img: PIL.Image.Image, target: dict):
        w, h = target["size"]
        bboxes = [
            bbox
            for label, bbox in zip(target["labels"], target["boxes"])
            if label.item() in self.labels
        ]
        if len(bboxes) > 0:
            object_num = random.randint(0, len(bboxes) - 1)
            left = random.randint(0, self.left_max_pad)
            top = random.randint(0, self.top_max_pad)
            right = random.randint(0, self.right_max_pad)
            bottom = random.randint(0, self.bottom_max_pad)
            bbox = bboxes[object_num].tolist()
            # target["crop_orig_size"] = torch.tensor([bbox[3]-bbox[1]+y_margin*2, bbox[2]-bbox[0]+x_margin*2])
            # target["crop_orig_offset"] = torch.tensor([bbox[0]-x_margin, bbox[1]-y_margin])
            region = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            # transpose and add margin
            region = [
                region[1] - top,
                region[0] - left,
                region[3] + top + bottom,
                region[2] + left + right,
            ]
            region = [round(elem) for elem in region]
            return R.crop(img, target, region)
        else:
            return img, target


class RandomCrop(object):
    def __init__(self, prob, left_pixels, top_pixels, right_pixels, bottom_pixels):
        self.prob = prob
        self.left_pixels = left_pixels
        self.top_pixels = top_pixels
        self.right_pixels = right_pixels
        self.bottom_pixels = bottom_pixels

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, height = image.size
            left = random.randint(0, self.left_pixels)
            top = random.randint(0, self.top_pixels)
            right = random.randint(0, self.right_pixels)
            bottom = random.randint(0, self.bottom_pixels)
            cropped_image = image.crop((left, top, right, bottom))
            cropped_bboxes = []
            cropped_labels = []
            for bbox, label in zip(target["boxes"], target["labels"]):
                bbox = [
                    max(bbox[0], left) - left,
                    max(bbox[1], top) - top,
                    min(bbox[2], right) - left,
                    min(bbox[3], bottom) - top,
                ]
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    cropped_bboxes.append(bbox)
                    cropped_labels.append(label)

            if len(cropped_bboxes) > 0:
                target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(cropped_labels, dtype=torch.int64)
                return cropped_image, target

        return image, target


class RandomPercentageCrop(object):
    def __init__(self, prob, left_scale, top_scale, right_scale, bottom_scale):
        self.prob = prob
        self.left_scale = left_scale
        self.top_scale = top_scale
        self.right_scale = right_scale
        self.bottom_scale = bottom_scale

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, height = image.size
            left = int(math.floor(width * 0.5 * self.left_scale * random.random()))
            top = int(math.floor(height * 0.5 * self.top_scale * random.random()))
            right = width - int(
                math.floor(width * 0.5 * self.right_scale * random.random())
            )
            bottom = height - int(
                math.floor(height * 0.5 * self.bottom_scale * random.random())
            )
            cropped_image = image.crop((left, top, right, bottom))
            cropped_bboxes = []
            cropped_labels = []
            for bbox, label in zip(target["boxes"], target["labels"]):
                bbox = [
                    max(bbox[0], left) - left,
                    max(bbox[1], top) - top,
                    min(bbox[2], right) - left,
                    min(bbox[3], bottom) - top,
                ]
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    cropped_bboxes.append(bbox)
                    cropped_labels.append(label)

            if len(cropped_bboxes) > 0:
                target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(cropped_labels, dtype=torch.int64)
                return cropped_image, target

        return image, target


class ColorJitterWithTarget(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, img: PIL.Image.Image, target: dict):
        img = self.transform(img)

        return img, target


class RandomErasingWithTarget(object):
    def __init__(
        self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=255, inplace=False
    ):
        self.transform = transforms.RandomErasing(
            p=p, scale=scale, ratio=ratio, value=value, inplace=False
        )

    def __call__(self, img: PIL.Image.Image, target: dict):
        img = self.transform(img)

        return img, target


class ToPILImageWithTarget(object):
    def __init__(self):
        self.transform = transforms.ToPILImage()

    def __call__(self, img: PIL.Image.Image, target: dict):
        img = self.transform(img)

        return img, target


class RandomDilation(object):
    def __init__(self, probability=0.5, size=3):
        self.probability = probability
        self.filter = ImageFilter.RankFilter(
            size, int(round(0 * size * size))
        )  # 0 is equivalent to a min filter

    def __call__(self, img: PIL.Image.Image, target: dict):
        r = random.random()

        if r <= self.probability:
            img = img.filter(self.filter)

        return img, target


class RandomErosion(object):
    def __init__(self, probability=0.5, size=3):
        self.probability = probability
        self.filter = ImageFilter.RankFilter(
            size, int(round(0.6 * size * size))
        )  # Almost a median filter

    def __call__(self, img: PIL.Image.Image, target: dict):
        r = random.random()

        if r <= self.probability:
            img = img.filter(self.filter)

        return img, target


class RandomResize(object):
    def __init__(self, min_min_size, max_min_size, max_max_size):
        self.min_min_size = min_min_size
        self.max_min_size = max_min_size
        self.max_max_size = max_max_size

    def __call__(self, image, target):
        width, height = image.size
        current_min_size = min(width, height)
        current_max_size = max(width, height)
        min_size = random.randint(self.min_min_size, self.max_min_size)
        if current_max_size * min_size / current_min_size > self.max_max_size:
            scale = self.max_max_size / current_max_size
        else:
            scale = min_size / current_min_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )
        resized_bboxes = []
        for bbox in target["boxes"]:
            bbox = [scale * elem for elem in bbox]
            resized_bboxes.append(bbox)

        target["boxes"] = torch.as_tensor(resized_bboxes, dtype=torch.float32)

        return resized_image, target


class RandomMaxResize(object):
    def __init__(self, min_max_size, max_max_size):
        self.min_max_size = min_max_size
        self.max_max_size = max_max_size

    def __call__(self, image, target):
        width, height = image.size
        current_max_size = max(width, height)
        target_max_size = random.randint(self.min_max_size, self.max_max_size)
        scale = target_max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )
        resized_bboxes = []
        for bbox in target["boxes"]:
            bbox = [scale * elem for elem in bbox]
            resized_bboxes.append(bbox)

        target["boxes"] = torch.as_tensor(resized_bboxes, dtype=torch.float32)

        return resized_image, target


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class PDFTablesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        transforms=None,
        max_size=None,
        do_crop=True,
        make_coco=False,
        include_original=False,
        max_neg=None,
        negatives_root=None,
        xml_fileset="filelist.txt",
        image_extension=".png",
        class_map=None,
    ):
        self.root = root
        self.transforms = transforms
        self.do_crop = do_crop
        self.make_coco = make_coco
        self.image_extension = image_extension
        self.class_map = class_map
        self.class_list = list(class_map)
        self.class_set = set(class_map.values())
        self.class_set.remove(class_map["no object"])

        try:
            with open(os.path.join(root, "..", xml_fileset), "r") as file:
                lines = file.readlines()
                lines = [l.split("/")[-1] for l in lines]
        except:
            lines = os.listdir(root)
        xml_page_ids = set(
            [f.strip().replace(".xml", "") for f in lines if f.strip().endswith(".xml")]
        )

        image_directory = os.path.join(root, "..", "images")
        try:
            with open(os.path.join(image_directory, "filelist.txt"), "r") as file:
                lines = file.readlines()
        except:
            lines = os.listdir(image_directory)
        png_page_ids = set(
            [
                f.strip().replace(self.image_extension, "")
                for f in lines
                if f.strip().endswith(self.image_extension)
            ]
        )

        self.page_ids = list(xml_page_ids.intersection(png_page_ids))
        if not max_size is None:
            self.page_ids = random.sample(self.page_ids, max_size)
        num_page_ids = len(self.page_ids)
        self.types = [1 for idx in range(num_page_ids)]

        if not max_neg is None and max_neg > 0:
            with open(os.path.join(negatives_root, "filelist.txt"), "r") as file:
                neg_xml_page_ids = set(
                    [
                        f.strip().replace(".xml", "")
                        for f in file.readlines()
                        if f.strip().endswith(".xml")
                    ]
                )
                neg_xml_page_ids = neg_xml_page_ids.intersection(png_page_ids)
                neg_xml_page_ids = list(neg_xml_page_ids.difference(set(self.page_ids)))
                if len(neg_xml_page_ids) > max_neg:
                    neg_xml_page_ids = random.sample(neg_xml_page_ids, max_neg)
            self.page_ids += neg_xml_page_ids
            self.types += [0 for idx in range(len(neg_xml_page_ids))]

        self.has_mask = False
        self.include_original = include_original

        if self.make_coco:
            self.dataset = {}
            self.dataset["images"] = [
                {"id": idx} for idx, _ in enumerate(self.page_ids)
            ]
            self.dataset["annotations"] = []
            ann_id = 0
            for image_id, page_id in enumerate(self.page_ids):
                annot_path = os.path.join(self.root, page_id + ".xml")
                bboxes, labels = read_pascal_voc(annot_path, class_map=self.class_map)

                # Reduce class set
                keep_indices = [
                    idx for idx, label in enumerate(labels) if label in self.class_set
                ]
                bboxes = [bboxes[idx] for idx in keep_indices]
                labels = [labels[idx] for idx in keep_indices]

                for bbox, label in zip(bboxes, labels):
                    ann = {
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        "iscrowd": 0,
                        "bbox": [
                            bbox[0],
                            bbox[1],
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                        ],
                        "category_id": label,
                        "image_id": image_id,
                        "id": ann_id,
                        "ignore": 0,
                        "segmentation": [],
                    }
                    self.dataset["annotations"].append(ann)
                    ann_id += 1
            self.dataset["categories"] = [{"id": idx} for idx in self.class_list[:-1]]

            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        print("index created!")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def __getitem__(self, idx):
        # load images ad masks
        page_id = self.page_ids[idx]
        img_path = os.path.join(
            self.root, "..", "images", page_id + self.image_extension
        )
        annot_path = os.path.join(self.root, page_id + ".xml")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if self.types[idx] == 1:
            bboxes, labels = read_pascal_voc(annot_path, class_map=self.class_map)

            # Reduce class set
            keep_indices = [
                idx for idx, label in enumerate(labels) if label in self.class_set
            ]
            bboxes = [bboxes[idx] for idx in keep_indices]
            labels = [labels[idx] for idx in keep_indices]

            # Convert to Torch Tensor
            if len(labels) > 0:
                bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
            else:
                # Not clear if it's necessary to force the shape of bboxes to be (0, 4)
                bboxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        num_objs = bboxes.shape[0]

        # Create target
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = bboxes[:, 2] * bboxes[:, 3]  # COCO area
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.transforms is not None:
            img_tensor, target = self.transforms(img, target)

        if self.include_original:
            return img_tensor, target, img, img_path
        else:
            return img_tensor, target

    def __len__(self):
        return len(self.page_ids)

    def getImgIds(self):
        return range(len(self.page_ids))

    def getCatIds(self):
        return range(10)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                lists = [
                    self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(catIds) == 0
                else [ann for ann in anns if ann["category_id"] in catIds]
            )
            anns = (
                anns
                if len(areaRng) == 0
                else [
                    ann
                    for ann in anns
                    if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
                ]
            )

            ids = [ann["id"] for ann in anns]
        return ids
