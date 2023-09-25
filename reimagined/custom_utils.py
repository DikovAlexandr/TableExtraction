import matplotlib.pyplot as plt

def save_mAP(OUT_DIR, map_05, map, iou_type='bbox'):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.

    :param OUT_DIR: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU.
    :param map: List containing mAP values at 0.5:0.95 IoU.
    :param name: Type of plot, bbox or segm.
    """

     # Create a new figure for the plot and add a subplot
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()

    # Plot mAP@0.5 values in orange
    ax.plot(
        map_05, color='tab:orange', linestyle='-', 
        label=f"mAP_{iou_type}_@0.50"
    )

    # Plot mAP@0.5:0.95 values in red
    ax.plot(
        map, color='tab:red', linestyle='-', 
        label=f"mAP_{iou_type}_@0.50:0.95"
    )
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel(iou_type+'_mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/{iou_type}_map.png")