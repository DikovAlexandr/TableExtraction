"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher:
    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
"""

import datetime
import os
import time
import torch.nn as nn
import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from metrics import utils
from metrics.coco_utils import get_coco, get_coco_kp
from engine import evaluate, train_one_epoch
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste
from class_names import INSTANCE_CATEGORY_NAMES as class_names
from custom_utils import save_mAP

def copypaste_collate_fn(batch):
    """
    Function for copy-paste augmentation.
    Args:
        batch (list): A list augmentations in batch.
    Returns:
        The result of SimpleCopyPaste.
    """
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def get_dataset(name, image_set, transform, data_path):
    """
    Retrieves a dataset based on the given name, image set, transformation, and data path.
    Args:
        name (str): The name of the dataset to retrieve.
        image_set (str): The image set to use for the dataset.
        transform (callable): A function or transform to apply to the dataset.
        data_path (str): The path to the dataset.
    Returns:
        tuple: A tuple containing the dataset and the number of classes in the dataset.
    """
    paths = {"coco": (data_path, get_coco, 91), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, args):
    """
    Function to get the appropriate data transformation based on training mode and arguments.
    Args:
        train (bool): A flag indicating whether the transform is for training or not.
        args (Namespace): An object containing the command line arguments.
    Returns:
        presets.DetectionPresetTrain or presets.DetectionPresetEval or callable.
    """
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()


# Function to get the works settings 
def get_args_parser(add_help=True):
    """
    Creates an argument parser for command line arguments.
    Args:
        add_help (bool, optional): Whether to include the help option in the parser. Defaults to True.
    Returns:
        argparse.ArgumentParser: The argument parser object.
    """
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    return parser


def main(args):
    # Create the output directory if specified
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # Initialize distributed mode for multi-GPU training
    utils.init_distributed_mode(args)
    print(args)

    # Set the device
    try:
        # Try to use the device specified in the arguments.
        device = torch.device(args.device)
        torch.randn(1).to(device)  # A test operation to check if the specified device works.
    except Exception as e:
        # If an error occurs, switch to 'cuda' if available, otherwise use 'cpu'.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     # Use deterministic algorithms if specified
    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Load the dataset
    print("Loading data...")
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(True, args), args.data_path)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(False, args), args.data_path)
    print("Dataset loaded.")

    # Create data loaders
    print("Creating data loaders...")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    # Use CopyPaste collate function if specified
    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    # Create data loaders for training and testing
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    print("Data loaders created.")

    # Create the model
    print("Creating model...")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    model = torchvision.models.get_model(
        args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, **kwargs
    )

    # Customize certain model components
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
    model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))

    # Print detailed model information
    print(model)
    print('Model created.')

    # Calculate and print total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Move the model to the specified device
    model.to(device)

    # Convert model to DistributedDataParallel if distributed training is enabled
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Define optimization parameters
    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    # Define the optimizer based on the chosen optimizer
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    # Use mixed precision training if specified
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Define the learning rate scheduler based on the specified arguments
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        # If resuming training from a checkpoint, load the saved model, optimizer, and lr_scheduler states
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # If the test-only mode is enabled, perform model evaluation and return
        torch.backends.cudnn.deterministic = True
        evaluate(model, data_loader_test, device=device)
        return
    
    # Initialize lists to store validation mAP scores at different IoU thresholds
    val_map_05_bbox = []
    val_map_bbox = []
    val_map_05_segm = []
    val_map_segm = []

    # Start the training loop
    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Train the model for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        lr_scheduler.step()

        if args.output_dir:
            # Save a checkpoint of the model, optimizer, and lr_scheduler
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # Evaluate the model after each epoch and collect mAP scores
        stats = evaluate(model, data_loader_test, device=device)
        val_map_05_bbox.append(stats['bbox'][1])
        val_map_bbox.append(stats['bbox'][0])
        val_map_05_segm.append(stats['segm'][1])
        val_map_segm.append(stats['segm'][0])

        # Save mAP plots for visualization
        save_mAP(args.output_dir, val_map_05_bbox, val_map_bbox, iou_type='bbox')
        save_mAP(args.output_dir, val_map_05_segm, val_map_segm, iou_type='segm')

    # Calculate and print the total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)