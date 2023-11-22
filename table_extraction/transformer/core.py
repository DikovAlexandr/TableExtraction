"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
from datetime import datetime
import json
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import cv2
from PIL import Image

sys.path.append(os.path.join('transformer', 'detr'))
sys.path.append(os.path.join('transformer', 'src'))

from transformer.detr.engine import evaluate, train_one_epoch
from transformer.detr.models import build_model
import transformer.detr.util.misc as utils
import transformer.detr.datasets.transforms as R

# from config import Args
# from table_datasets import (
#     PDFTablesDataset,
#     TightAnnotationCrop,
#     RandomPercentageCrop,
#     RandomErasingWithTarget,
#     ToPILImageWithTarget,
#     RandomMaxResize,
#     RandomCrop,
# )


def get_class_map():
    """
    Returns a dictionary representing a class mapping.
    The class mapping is a dictionary where each key represents a class name and each value represents the corresponding class ID.
    Returns:
        dict: A dictionary representing the class mapping.
    """
    class_map = {
        "table": 0,
        "table column": 1,
        "table row": 2,
        "table column header": 3,
        "table projected row header": 4,
        "table spanning cell": 5,
        "no object": 6,
    }
    return class_map


def get_model(args, device):
    """
    Loads DETR model on to the device specified.
    If a load path is specified, the state dict is updated accordingly.
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    if args.model_load_path:
        # print("loading model from checkpoint")
        loaded_state_dict = torch.load(args.model_load_path, map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=True)
    return model, criterion, postprocessors


def load_json(json_path):
    data = None
    with open(json_path) as ref:
        data = json.load(ref)
    return data


class TableRecognizer:
    def __init__(self, checkpoint_path, mode):
        if mode == 'detection':
            config = 'detection_config.json'
        elif mode == 'structure':
            config = 'structure_config.json'
        else:
            print ("Invalid mode")
            config = 'structure_config.json'
        
        args = load_json(os.path.join(os.getcwd(), 'transformer', 'src', config))
        args = type("Args", (object,), args)

        assert os.path.exists(checkpoint_path), checkpoint_path

        args.model_load_path = checkpoint_path

        # Fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # print("loading model")
        self.device = torch.device(args.device)
        self.model, _, self.postprocessors = get_model(args, self.device)
        self.model.eval()

        class_map = get_class_map()

        self.normalize = R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def predict(self, image_path=None, debug=True, thresh=0.9):
        image = image_path

        w, h, _ = image.shape

        img_tensor = self.normalize(F.to_tensor(image))[0]
        img_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)

        outputs = None
        with torch.no_grad():
            outputs = self.model(img_tensor)

        image_size = torch.unsqueeze(torch.as_tensor([int(h), int(w)]), 0).to(
            self.device
        )
        results = self.postprocessors["bbox"](outputs, image_size)[0]

        if debug is True:
            image = np.array(image)
            for idx, score in enumerate(results["scores"].tolist()):
                if score < thresh:
                    continue

                xmin, ymin, xmax, ymax = list(map(int, results["boxes"][idx]))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            results["debug_image"] = image

        return results
