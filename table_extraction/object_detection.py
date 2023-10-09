import os
import cv2
import logging
import itertools
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from maskrcnn import inference

logging.basicConfig(
    filename='logs.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TESSERACT_CONFIG = r'--oem 1 --psm 1 -l rus'

def get_tables_maskrcnn(low_quality_gray_images: List[np.ndarray], 
                         low_dpi: int, 
                         high_quality_gray_images: List[np.ndarray], 
                         high_dpi: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extracts tables from low and high-quality grayscale images using Mask R-CNN.

    Args:
        low_quality_gray_images (List[np.ndarray]): List of low-quality grayscale images.
        low_dpi (int): DPI of the low-quality images.
        high_quality_gray_images (List[np.ndarray]): List of high-quality grayscale images.
        high_dpi (int): DPI of the high-quality images.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
        - List of extracted high-quality table images.
        - List of extracted low-quality table images.
    """
    tables_low = []
    tables_high =[]

    for num, image in enumerate(low_quality_gray_images):
        # Get prediction
        mode = "table_plot"
        weights = os.getcwd() + "\maskrcnn\weights\table_plot.pth"
        _, boxes, labels = inference.get_bboxes_of_objects(image, weights, threshold = 0.8, mode=mode)

        logging.debug(f"Image {num + 1} - Boxes: {boxes} [amount {len(boxes)}], Labels: {labels} [amount {len(labels)}]")
        
        for box, label in zip(boxes, labels):
            if label == "table":
                [x1, y1], [x2, y2] = box
                croped_low_quality_gray_image = low_quality_gray_images[num][y1:y2, x1:x2]
                tables_low.append(croped_low_quality_gray_image)
                croped_high_quality_gray_image = high_quality_gray_images[num][y1:y2, x1:x2]
                tables_high.append(croped_high_quality_gray_image)
        return tables_high, tables_low
    
def get_cells_maskrcnn(tables: List[np.ndarray]) -> List[List[Tuple[int, int, int, int]]]:
    """
    Extracts cells from tables using Mask R-CNN.

    Args:
        tables (List[np.ndarray]): List of table images.

    Returns:
        List[List[Tuple[int, int, int, int]]]: List of cell bounding boxes for each table.
    """
    all_cells = []

    for num, image in enumerate(tables):
        cells = []
        copy_image = image.copy()

        # Get prediction
        mode = "cells"
        weights = os.getcwd() + "\maskrcnn\weights\cell_detection.pth"
        _, boxes, labels = inference.get_bboxes_of_objects(copy_image, weights, threshold=0.8, mode=mode)

        logging.debug(f"Image {num + 1} - Boxes: {boxes} [amount {len(boxes)}], Labels: {labels} [amount {len(labels)}]")

        for box, _ in zip(boxes, labels):
            [x1, y1], [x2, y2] = box
            cells.append((x1, y1, x2, y2))

            cv2.rectangle(copy_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        plt.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        all_cells.append(cells)
    return(all_cells)


def visualize_table_images(tables: List[np.ndarray]) -> None:
    """
    Visualizes a list of table images.

    Args:
        tables (List[np.ndarray]): List of table images.

    Returns:
        None
    """
    for table in tables:
        plt.imshow(cv2.cvtColor(table, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def get_lines_Hough(tables: List[np.ndarray]) -> List[Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]]:
    """
    Extracts vertical and horizontal lines from a list of table images using Hough Line Transform.

    Args:
        tables (List[np.ndarray]): List of table images.

    Returns:
        List[Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]]:
        - List of vertical lines represented as (x1, y1, x2, y2) tuples for each table.
        - List of horizontal lines represented as (x1, y1, x2, y2) tuples for each table.
    """
    tables_lines = []

    # Extract vertical and horizontal lines in an image using a kernel transform
    for num, image in enumerate(tables):
        height, width = image.shape
        _, threshold_image = cv2.threshold(
            image, 127, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(threshold_image)

        # Horizontal
        hor = np.array([[1, 1, 1, 1, 1, 1]])
        vertical_lines_eroded_image = cv2.erode(
            inverted_image, hor, iterations=10)
        vertical_lines_eroded_image = cv2.dilate(
            vertical_lines_eroded_image, hor, iterations=10)

        # Vertical
        ver = np.array([[1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1]])
        horizontal_lines_eroded_image = cv2.erode(
            inverted_image, ver, iterations=10)
        horizontal_lines_eroded_image = cv2.dilate(
            horizontal_lines_eroded_image, ver, iterations=10)

        # Combine
        combined_image = cv2.add(
            vertical_lines_eroded_image, horizontal_lines_eroded_image)

        # Dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        combined_image_dilated = cv2.dilate(
            combined_image, kernel, iterations=5)

        lines = cv2.HoughLinesP(
            combined_image_dilated, 1, np.pi / 180, 50, None, 50, 10)

        # Recurrent function for checking structures for emptiness
        def is_not_empty_element(structure):
            if isinstance(structure, list) or isinstance(structure, np.ndarray):
                return any(is_not_empty_element(item) for item in structure)
            elif isinstance(structure, int) or isinstance(structure, np.intc):
                return True
            else:
                return False

        # Finding all vertical and horizontal lines of a table
        vertical_lines = []
        horizontal_lines = []

        if is_not_empty_element(lines):
            borders = np.concatenate(lines, axis=0)

            # Tolerance parameters at which we consider that the lines are even and intersect
            epsilon = (height + width) * 0.01
            for x1, y1, x2, y2 in borders:
                if abs(x1 - x2) <= epsilon and abs(y1 - y2) > epsilon:  # Vertical lines
                    vertical_lines.append(
                        (int((x1 + x2) / 2), height - y1, int((x1 + x2) / 2), height - y2))
                elif abs(y1 - y2) <= epsilon and abs(x1 - x2) > epsilon:  # Horizontal lines
                    horizontal_lines.append(
                        (int(x1), height - int((y1 + y2) / 2), int(x2), height - int((y1 + y2) / 2)))

            # Save lines for visualisation
            tables_lines.append((vertical_lines, horizontal_lines))
        else:
            tables_lines.append(([], []))
    return tables_lines

def visualize_tables_lines(tables: List[np.ndarray], 
                           tables_lines: List[Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]]) -> None:
    """
    Visualizes the detected lines in a list of table images.

    Args:
        tables (List[np.ndarray]): List of table images.
        tables_lines (List[Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]]): 
            List of detected vertical and horizontal lines for each table.

    Returns:
        None
    """
    for num, table in enumerate(tables):
        copy_image = table.copy()
        height, _ = copy_image.shape
        vertical_lines, horizontal_lines = tables_lines[num]
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.rectangle(copy_image, (x1, height - y1),
                            (x2, height - y2), (0, 255, 0), 5)
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.rectangle(copy_image, (x1, height - y1),
                            (x2, height - y2), (0, 255, 0), 5)

        plt.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def get_nodes(tables: List[np.ndarray], 
              tables_lines: List[Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]]) -> List[List[Tuple[int, int]]]:
    """
    Extracts and sorts the coordinates of table nodes based on detected table lines.

    Args:
        tables (List[np.ndarray]): List of table images.
        tables_lines (List[Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]]): 
            List of detected vertical and horizontal lines for each table.

    Returns:
        List[List[Tuple[int, int]]]: List of table node coordinates for each table.
    """
    all_tables_nodes = []
    # Finding the coordinates of table nodes
    for num, image in enumerate(tables):
        height, width = image.shape
        epsilon = (height + width) * 0.01
        table_nodes = []
        nodes_sorted_xy = []
        if tables_lines[num][0] and tables_lines[num][1]:
            for v_line, h_line in itertools.product(tables_lines[num][0], tables_lines[num][1]):
                v_x1, v_y1, v_x2, v_y2 = v_line
                h_x1, h_y1, h_x2, h_y2 = h_line

                # Checking if there are lines or whether they end in the neighborhood
                if (((h_x1 - epsilon <= v_x1 <= h_x2 + epsilon) and
                    (v_y1 - epsilon <= h_y1 <= v_y2 + epsilon)) or
                    (abs(h_x1 - v_x1) <= epsilon and v_y1 - epsilon <= h_y1 <= v_y2 + epsilon) or
                    (abs(h_x2 - v_x1) <= epsilon and v_y1 - epsilon <= h_y1 <= v_y2 + epsilon) or
                    (abs(h_y1 - v_y1) <= epsilon and h_x1 - epsilon <= v_x1 <= h_x2 + epsilon) or
                        (abs(h_y1 - v_y2) <= epsilon and h_x1 - epsilon <= v_x1 <= h_x2 + epsilon)):
                    table_nodes.append((v_x1, h_y1))

                if v_x1 <= epsilon or v_y1 <= epsilon or abs(v_x1 - width) <= epsilon or abs(v_y1 - height) <= epsilon:
                    table_nodes.append((v_x1, v_y1))

                if v_x2 <= epsilon or v_y2 <= epsilon or abs(v_x2 - width) <= epsilon or abs(v_y2 - height) <= epsilon:
                    table_nodes.append((v_x2, v_y2))

                if h_x1 <= epsilon or h_y1 <= epsilon or abs(h_x1 - width) <= epsilon or abs(h_y1 - height) <= epsilon:
                    table_nodes.append((h_x1, h_y1))

                if h_x2 <= epsilon or h_y2 <= epsilon or abs(h_x2 - width) <= epsilon or abs(h_y2 - height) <= epsilon:
                    table_nodes.append((h_x2, h_y2))

                if v_y1 <= epsilon or v_y2 <= epsilon:
                    table_nodes.append((0, 0))
                    table_nodes.append((width, 0))

            if table_nodes == []:
                all_tables_nodes.append([])
                break

            # Create a copy of the table_nodes list for modification
            modified_table_nodes = np.array(table_nodes.copy())

            # Looking for points that are in the neighborhood of each other using KDTree
            kdtree = KDTree(modified_table_nodes)

            neighborhood_nodes = []
            visited = set()

            for node in modified_table_nodes:
                if tuple(node) in visited:
                    continue

                idxs = kdtree.query_ball_point(node, epsilon)
                visited.update(
                    tuple(modified_table_nodes[i]) for i in idxs)

                if len(idxs) > 1:
                    mean_node = np.round(
                        np.mean(modified_table_nodes[idxs], axis=0)).astype(int)
                    neighborhood_nodes.append(tuple(mean_node))
                else:
                    neighborhood_nodes.append(tuple(node))

            # Sort nodes by x and y axis
            nodes_sorted_x = sorted(neighborhood_nodes, key=lambda x: x[0])

            for i in range(len(nodes_sorted_x)-1):
                if abs(nodes_sorted_x[i][0] - nodes_sorted_x[i+1][0]) <= epsilon:
                    nodes_sorted_x[i+1] = (nodes_sorted_x[i]
                                            [0], nodes_sorted_x[i+1][1])

            nodes_sorted_y = sorted(nodes_sorted_x, key=lambda x: x[1])

            for i in range(len(nodes_sorted_y)-1):
                if abs(nodes_sorted_y[i][1] - nodes_sorted_y[i+1][1]) <= epsilon:
                    nodes_sorted_y[i+1] = (nodes_sorted_y[i+1]
                                            [0], nodes_sorted_y[i][1])

            nodes_sorted_xy = sorted(
                nodes_sorted_y, key=lambda x: (-x[1], x[0]))

            # Remove the same coordinates of points until the length becomes constant
            while True:
                count_x = {}
                count_y = {}
                for x, y in nodes_sorted_xy:
                    count_x[x] = count_x.get(x, 0) + 1
                    count_y[y] = count_y.get(y, 0) + 1

                filtered_points = [
                    (x, y) for x, y in nodes_sorted_xy if count_x[x] > 1 and count_y[y] > 1]
                # tresh_points = [(x, y) for x, y in nodes_sorted_xy if count_x[x] <= 1 or count_y[y] <= 1]

                if len(filtered_points) == len(nodes_sorted_xy):
                    break
                else:
                    nodes_sorted_xy = filtered_points

        all_tables_nodes.append(nodes_sorted_xy)
    return all_tables_nodes

def visualize_tables_nodes(tables: List[np.ndarray], 
                           tables_nodes: List[List[Tuple[int, int]]]) -> None:
    """
    Visualizes the detected table nodes in a list of table images.

    Args:
        tables (List[np.ndarray]): List of table images.
        tables_nodes (List[List[Tuple[int, int]]]): List of detected table node coordinates for each table.

    Returns:
        None
    """
    for num, table in enumerate(tables):
        copy_image = table.copy()
        height, _ = copy_image.shape
        if tables_nodes[num]:
            for node in tables_nodes[num]:
                x, y = node
                cv2.rectangle(copy_image, (x, height - y),
                                (x + 10, height - y + 10), (0, 255, 0), 10)

        plt.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def get_rectangles(tables: List[np.ndarray], 
                   tables_nodes: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int, int, int]]]:
    """
    Extracts rectangles within tables based on nodes.

    Args:
        tables (List[np.ndarray]): List of table images.
        tables_nodes (List[List[Tuple[int, int]]]): List of nodes for each table image.

    Returns:
        List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        - List of rectangles for each table image. Each rectangle is represented as a tuple of two points.
          The first point is the top-left corner, and the second point is the bottom-right corner.
    """
    # Initialize a list to store the rectangles for each table
    tables_rectangles = []

    # Loop through each table image and its corresponding nodes
    for num, image in enumerate(tables):
        height, width = image.shape
        epsilon = (height + width) * 0.01
        tables_nodes = tables_nodes[num]
        
        # Create a list of rectangles
        rectangles = []
        for i in range(len(tables_nodes) - 1):
            current_node = tables_nodes[i]
            if tables_nodes[i + 1][1] == current_node[1]:
                next_x_node = tables_nodes[i + 1]
            else:
                continue

            # Find the next nodes along the y-axis for the current nodes
            next_y_nodes = [node for node in tables_nodes if abs(
                node[0] - current_node[0]) <= epsilon and node[1] < current_node[1]]
            flag = True
            for next_y_node in next_y_nodes:
                opposite_node = (next_x_node[0], next_y_node[1])
                if flag:
                    for node in tables_nodes:
                        if abs(node[0] - opposite_node[0]) <= epsilon and abs(node[1] - opposite_node[1]) <= epsilon:

                            # Build a rectangle based on the found nodes
                            rectangle = (
                                current_node[0], current_node[1], opposite_node[0], opposite_node[1])
                            rectangles.append(rectangle)
                            flag = False
                            break

        rectangles = sorted(rectangles, key=lambda x: (- x[0][1], x[0][0]))
        tables_rectangles.append(rectangles)
    return tables_rectangles

def visualize_rectangles(tables: List[np.ndarray],
                         tables_rectangles: List[List[Tuple[int, int, int, int]]]) -> None:
    """
    Visualizes a list of rectangles on tables.

    Args:
        rectangles (List[List[Tuple[int, int, int, int]]]): A list of lists of rectangle coordinates.
            Each inner list contains tuples (x1, y1, x2, y2) representing the top-left and bottom-right
            corners of rectangles on images.

    Returns:
        None
    """
    for image, rectangles in zip(tables, tables_rectangles):
        copy_image = image.copy()
        for rectangle in rectangles:
            x1, y1, x2, y2 = rectangle

            # Draw rectangles on the image
            cv2.rectangle(copy_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the image with rectangles
        plt.imshow(copy_image)
        plt.axis('off')
        plt.show()