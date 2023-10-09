import cv2
import numpy as np
import torch

from . import class_names

np.random.seed(2023)

def get_outputs(image, model, threshold, mode):
    if mode == 'cells':
        coco_names = class_names.CELLS_CATEGORY_NAMES
    else:
        coco_names = class_names.INSTANCE_CATEGORY_NAMES

    with torch.no_grad():
        # Forward pass of the image through the model.
        outputs = model(image)
    
    # Get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # Index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # Get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # Discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]

    # Get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # Discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # Get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    # Discard bounding boxes below threshold value
    labels = labels[:thresholded_preds_count]
    return masks, boxes, labels    

def draw_segmentation_map(image, masks, boxes, labels, args, mode):
    if mode == 'cells':
        coco_names = class_names.CELLS_CATEGORY_NAMES
    else:
        coco_names = class_names.INSTANCE_CATEGORY_NAMES
    # This will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

    alpha = 1.0
    beta = 1.0 # Transparency for the segmentation map
    gamma = 0.0 # Scalar added to each sum
    # Convert the original PIL image into NumPy format
    image = np.array(image)
    # Convert from RGN to OpenCV BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(len(masks)):
        # Apply a randon color mask to each object
        color = COLORS[coco_names.index(labels[i])]
        if masks[i].any() == True:
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
            # Combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            # Apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

            lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
            tf = max(lw - 1, 1) # Font thickness.
            p1, p2 = boxes[i][0], boxes[i][1]
            if not args.no_boxes:
                # Draw the bounding boxes around the objects
                cv2.rectangle(
                    image, 
                    p1, p2, 
                    color=color, 
                    thickness=lw,
                    lineType=cv2.LINE_AA
                )
                w, h = cv2.getTextSize(
                    labels[i], 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=lw / 3, 
                    thickness=tf
                )[0]  # Text width, height
                w = int(w - (0.20 * w))
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                # Put the label text above the objects
                cv2.rectangle(
                    image, 
                    p1, 
                    p2, 
                    color=color, 
                    thickness=-1, 
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    image, 
                    labels[i], 
                    (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=lw / 3.8, 
                    color=(255, 255, 255), 
                    thickness=tf, 
                    lineType=cv2.LINE_AA
                )
    return image