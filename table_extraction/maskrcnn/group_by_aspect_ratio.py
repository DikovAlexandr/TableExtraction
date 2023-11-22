import bisect
import copy
import math
from collections import defaultdict
from itertools import chain, repeat

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data.sampler import BatchSampler, Sampler
from torch.utils.model_zoo import tqdm


def _repeat_to_at_least(iterable, n):
    """
    Repeat an iterable to have at least `n` elements.
    Parameters:
        iterable (iterable): The iterable to be repeated.
        n (int): The desired minimum number of elements in the repeated iterable.
    Returns:
        list: A list containing the repeated elements from the original iterable.
    """
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Args:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}")
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        """
        Now we have run out of elements that satisfy the group criteria, let's return the remaining
        elements so that the size of the sampler is deterministic
        """

        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # For the remaining batches, take buffers with the largest number of elements
            for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size


def _compute_aspect_ratios_slow(dataset, indices=None):
    """
    Compute aspect ratios of images in a dataset.
    This function computes the aspect ratios of images in a dataset. If the dataset does not support a fast path for computing the aspect ratios, the function will iterate over the full dataset and load every image, which might take some time.
    Parameters:
    - dataset (torch.utils.data.Dataset): The dataset containing the images.
    - indices (list or None): The indices of the images to compute the aspect ratios for. If None, all images in the dataset will be used.
    Returns:
    - aspect_ratios (list): A list of aspect ratios for each image in the dataset.
    """
    print(
        "Your dataset doesn't support the fast path for "
        "computing the aspect ratios, so will iterate over "
        "the full dataset and load every image instead. "
        "This might take some time..."
    )
    if indices is None:
        indices = range(len(dataset))

    class SubsetSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    sampler = SubsetSampler(indices)

    # Create a DataLoader to iterate over the dataset
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=14,  # You might want to increase it for faster processing
        collate_fn=lambda x: x[0],
    )
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, (img, _) in enumerate(data_loader):
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    """
    Compute the aspect ratios for a custom dataset.
    Args:
        dataset (CustomDataset): The custom dataset object.
        indices (list, optional): A list of indices to compute aspect ratios for. Defaults to None.
    Returns:
        list: A list of aspect ratios for the given dataset and indices.
    """
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    """
    Compute the aspect ratios of images in a COCO dataset.
    Parameters:
        dataset (CocoDataset): The COCO dataset object.
        indices (list[int], optional): The indices of the images in the dataset to compute the aspect ratios for. If not specified, all images in the dataset will be used. Default is None.
    Returns:
        list[float]: A list of aspect ratios for the images in the dataset.
    """
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    """
    Compute the aspect ratios of images in a VOC dataset.
    Args:
        dataset (VocDataset): The VOC dataset.
        indices (list, optional): The indices of the images in the dataset to compute the aspect ratios for. Defaults to None, which computes the aspect ratios for all images in the dataset.
    Returns:
        list: A list of aspect ratios of the images.
    """
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    """
    Compute the aspect ratios of a subset of a dataset.
    Args:
        dataset (Dataset): The dataset containing the images.
        indices (List[int], optional): The indices of the subset to compute the aspect ratios for. If not provided, all indices will be used.
    Returns:
        List[float]: A list of aspect ratios for the subset of the dataset.
    """
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
    """
    Computes the aspect ratios for a given dataset.
    Args:
        dataset (torch.utils.data.Dataset): The dataset to compute aspect ratios for.
        indices (list, optional): A list of indices to compute aspect ratios for. 
            If None, computes aspect ratios for the entire dataset. Default is None.
    Returns:
        torch.Tensor: The computed aspect ratios.
    """
    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(dataset, indices)

    if isinstance(dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)

    # slow path
    return _compute_aspect_ratios_slow(dataset, indices)


def _quantize(x, bins):
    """
    Quantizes a list of values using a given set of bins.
    Parameters:
        x (List[float]): The list of values to be quantized.
        bins (List[float]): The list of bin edges.
    Returns:
        List[int]: The quantized values.
    Note:
        - The function uses deep copy of the `bins` list to avoid modifying the original list.
        - The `bins` list is sorted in ascending order.
        - The function uses `bisect.bisect_right` to find the index of the rightmost bin edge that is less than or equal to each value in `x`.
    """
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def create_aspect_ratio_groups(dataset, k=0):
    """
    Create aspect ratio groups based on the given dataset.
    Args:
        dataset (numpy.ndarray): The dataset to compute aspect ratios from.
        k (int, optional): The number of quantization bins. Defaults to 0.
    Returns:
        numpy.ndarray: The aspect ratio groups.
    """
    aspect_ratios = compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # Сount number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print(f"Using {fbins} as bins for aspect ratio quantization")
    print(f"Count of instances per bin: {counts}")
    return groups