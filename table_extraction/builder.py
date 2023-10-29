import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict

from maskrcnn import inference


def group_cells(cells: List[Tuple[int, int, int, int]], 
                epsilon: int = 20) -> Tuple[Dict[int, int], List[List[Tuple[int, int, int, int]]]]:
    """
    Group cells based on their y1 coordinates, where cells with y1 values differing by less than epsilon are considered part of the same group.

    Args:
        cells (List[Tuple[int, int, int, int]]): List of cell rectangles (x1, y1, x2, y2).
        epsilon (int): Maximum allowed difference in y1 values to group cells (default is 20).

    Returns:
        Tuple[Dict[int, int], List[List[Tuple[int, int, int, int]]]: A tuple containing a dictionary of grouped cell counts and a list of cell groups.
    """

    grouped_dict = defaultdict(int)
    grouped_list = []

    # Sort by y1
    cells.sort(key=lambda x: x[1])

    current_group = None
    current_group_list = []
    group_count = 0

    for item in cells:
        if current_group is None:
            current_group = item[1]
            group_count += 1
            current_group_list.append(item)
        elif abs(item[1] - current_group) <= epsilon:
            group_count += 1
            current_group_list.append(item)
        else:
            # New subgroup
            average_value = int(round(current_group))
            grouped_dict[average_value] = group_count
            grouped_list.append(current_group_list)
            current_group = item[1]
            group_count = 1
            current_group_list = [item]

    # Add last subgroup
    if current_group is not None:
        average_value = int(round(current_group))
        grouped_dict[average_value] = group_count
        grouped_list.append(current_group_list)

    return dict(grouped_dict), grouped_list


def split_into_headers_and_records(rectangles: List[Tuple[int, int, int, int]]) -> Tuple[List[Tuple[int, int, int, int]], 
                                                                                         List[Tuple[int, int, int, int]], 
                                                                                         List[List[Tuple[int, int, int, int]]]]:
    """
    Split cell rectangles into headers and records based on a change in the number of cells in a row.

    Args:
        rectangles (List[Tuple[int, int, int, int]]): List of cell rectangles (x1, y1, x2, y2).

    Returns:
        Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], List[List[Tuple[int, int, int, int]]]: A tuple containing header cells, record cells, and a list of cell groups for records.
    """

    cell_counts, _ = group_cells(rectangles)  # Dictionary to store cell counts for each y
    # print("cell_counts", cell_counts)

    # Find the y where the number of cells changes
    sorted_cell_counts = sorted(cell_counts.items(), reverse=True)
    # print("sorted_cell_counts", sorted_cell_counts)
    
    # Number of cells of each record line
    num_cells = None
    change_y = None
    epsilon = 20
    records = 1

    for y1, count in sorted_cell_counts:
        if num_cells is None:
            num_cells = count
        elif count != num_cells:
            change_y = y1
            break
        else:
            records += 1

    if change_y is None:
        change_y = sorted_cell_counts[-1][0]

    # print("change_y", change_y)

    record_cells = [(x1, y1, x2, y2)
                    for x1, y1, x2, y2 in rectangles if y1 >= change_y + epsilon]
    
    _, records_list = group_cells(record_cells)
    # print("records_list", records_list)
    
    record_cells = sorted(record_cells, key=lambda x: (x[1], x[0]))

    header_cells = [(x1, y1, x2, y2)
                    for x1, y1, x2, y2 in rectangles if y1 < change_y + epsilon]
    
    header_cells = sorted(header_cells, key=lambda x: (x[0], x[1]))

    return header_cells, record_cells, records_list


def split_into_headers_and_records_maskrcnn(table_image: np.ndarray,
                                            rectangles: List[Tuple[int, int, int, int]]) -> (Tuple[List[Tuple[int, int, int, int]], 
                                                                   List[Tuple[int, int, int, int]], int, int]):
    """
    Splits a list of rectangles into headers and records based on the number of cells per row using maskrcnn.

    Args:
        table_image (np.ndarray): The table image to analisation.
        rectangles (List[Tuple[int, int, int, int]]): A list of rectangles, where each rectangle is
        represented as a tuple of two points (top-left and bottom-right corners).

    Returns:
        Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], int, int]:
        - List of header rectangles, where each rectangle represents the header row.
        - List of record rectangles, where each rectangle represents a record row.
        - Number of cells per row (num_cells).
        - Total number of record rows (records).
    """
    copy_table_image = table_image.copy()
    header_cells = []
    record_cells = []

    # Get prediction
    mode = "cells"
    weights = os.path.join(os.getcwd(), "maskrcnn", "weights", "cell_classification.pth")
    _, boxes, labels = inference.get_bboxes_of_objects(table_image, weights, threshold=0.5, mode=mode)

    head_bbox = None
    record_bbox = None

    for box, label in zip(boxes, labels):
        [x1, y1], [x2, y2] = box
        if label == "head":
            if abs(x1 - x2)*abs(y1 - y2) >= max_head_area:
                head_bbox = (x1, y1, x2, y2)
                max_head_area = abs(x1 - x2)*abs(y1 - y2)
        elif label == "cell":
            if abs(x1 - x2)*abs(y1 - y2) >= max_record_area:
                record_bbox = (x1, y1, x2, y2)
                max_record_area = abs(x1 - x2)*abs(y1 - y2)

    if head_bbox:
        min_x_head, min_y_head, max_x_head, max_y_head = head_bbox
        # cv2.rectangle(copy_image, (min_x_head, min_y_head), (max_x_head, max_y_head), (0, 255, 0), 4)

    if record_bbox:
        min_x_record, min_y_record, max_x_record, max_y_record = record_bbox
        # cv2.rectangle(copy_image, (min_x_record, min_y_record), (max_x_record, max_y_record), (255, 0, 0), 4)
    
    # plt.imshow(copy_image)
    # plt.axis('off')
    # plt.show()

    for rectangle in rectangles:
        x1, y1, x2, y2 = rectangle

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if min_x_head <= center_x <= max_x_head and min_y_head <= center_y <= max_y_head:
            header_cells.append(rectangle)
        elif min_x_record <= center_x <= max_x_record and min_y_record <= center_y <= max_y_record:
            record_cells.append(rectangle)

    record_rows = []
    current_row = []
    prev_center_y = None
    epsilon = (table_image.shape[0] + table_image.shape[1])/ (2*40)

    for rectangle in record_cells:
        x1, y1, x2, y2 = rectangle
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if prev_center_y is None or abs(center_y - prev_center_y) <= epsilon:
            current_row.append(rectangle)
        else:
            record_rows.append(current_row)
            current_row = [rectangle]

        prev_center_y = center_y

    if current_row:
        record_rows.append(current_row)

    records = len(record_rows)
    num_cells = len(record_rows[0])

    return header_cells, record_cells, num_cells, records


def visualize_headers_and_records_cells(table_image: np.ndarray, 
                                        cells: List[Tuple[Tuple[int, int], Tuple[int, int]]], cell_type: str) -> None:
    """
    Visualizes header or record cells on an table image.

    Args:
        table_image (np.ndarray): The table image on which to visualize cells.
        cells (List[Tuple[Tuple[int, int], Tuple[int, int]]]): A list of cell rectangles, where each rectangle
        is represented as a tuple of two points (top-left and bottom-right corners).
        cell_type (str): Type of cells to visualize, either 'header' or 'record'.

    Returns:
        None
    """

    if cell_type == 'header':
        color = (120, 180, 0)
    elif cell_type == 'record':
        color = (0, 180, 180)
    copy_table_image = table_image.copy()
    copy_table_image = cv2.cvtColor(copy_table_image, cv2.COLOR_BGR2RGB)
    for cell in cells:
        x1, y1, x2, y2 = cell
        cv2.rectangle(copy_table_image, (x1, y1),
                        (x2, y2), color, 10)

    plt.imshow(cv2.cvtColor(copy_table_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def visualize_y_levels(table_image: np.ndarray,
                       cell_counts: Dict[int, int]) -> None:
    """
    Visualize horizontal lines at different y levels on an image and display cell counts next to them.

    Args:
        table_image: The input image.
        cell_counts: A dictionary containing cell counts for different y levels.

    Example:
        visualize_y_levels(table_image, cell_counts)
    """

    keys = list(cell_counts.keys())
    values = list(cell_counts.values())
    
    copy_table_image = table_image.copy()
    
    for i, key in enumerate(keys):
        x = 20 
        y = key
        
        cv2.line(copy_table_image, (x, y), (x + 10, y + 10), (0, 0, 0), 1)
        cv2.putText(copy_table_image, str(key) + " " + str(values[i]), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2)
    
    plt.imshow(cv2.cvtColor(copy_table_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def create_cell_dict(cells: List[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]]:
    """
    Creates a dictionary of overlapping cells.

    The function takes a list of cell rectangles and creates a dictionary in which the keys are the coordinates of
    each cell, and the values are lists of coordinates representing cells immediately below the corresponding key cell.

    Args:
        cells (List[Tuple[int, int, int, int]]): A list of cell rectangles, where each rectangle is represented as
        a tuple of four integers (x1, y1, x2, y2) defining the top-left and bottom-right corners of the cell.

    Returns:
        Dict[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]]: A dictionary where keys are cell coordinates
        and values are lists of coordinates of overlapping cells immediately below each key cell.
    """
    rectangle_dict = {}
    ignore_indices = []
    epsilon = 20

    for i, cell_1 in enumerate(cells):
        if i not in ignore_indices:
            overlapping_cells = []
            x1_cell_1, y1_cell_1, x2_cell_1, y2_cell_1 = cell_1

            for j, cell_2 in enumerate(cells):
                if i != j:
                    x1_cell_2, y1_cell_2, x2_cell_2, y2_cell_2 = cell_2
                    if (x1_cell_1 - epsilon <= x1_cell_2 < x2_cell_2 <= x2_cell_1 + epsilon) and abs(y2_cell_1 - y1_cell_2) <= epsilon and j not in ignore_indices:
                        overlapping_cells.append(cell_2)
                        ignore_indices.append(j)
            rectangle_dict[cell_1] = overlapping_cells

    return rectangle_dict


def visualize_cells_relationship(image: np.ndarray, 
                                 main_cell: Tuple[int, int, int, int], 
                                 overlapping_cells: List[Tuple[int, int, int, int]]) -> None:
    """
    Visualizes the relationship between a main cell and its overlapping cells on an image.

    Args:
        image (np.ndarray): The input image.
        main_cell (Tuple[int, int, int, int]): Coordinates of the main cell as a tuple (x1, y1, x2, y2).
        overlapping_cells (List[Tuple[int, int, int, int]]): A list of coordinates representing overlapping cells,
        where each cell is represented as a tuple (x1, y1, x2, y2).

    Returns:
        None
    """
    copy_image = image.copy()
    copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)

    x1, y1, x2, y2 = main_cell
    cv2.rectangle(copy_image, (x1, y1),
                    (x2, y2), (120, 180, 0), 10)

    for cell in overlapping_cells:
        x1, y1, x2, y2 = cell
        cv2.rectangle(copy_image, (x1, y1),
                        (x2, y2), (0, 180, 180), 10)

    plt.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def build_structure(cell: Tuple[int, int, int, int], 
                    rectangle_text_dict: dict, 
                    header_cell_dict: dict) -> dict:
    """
    Recursively builds a hierarchical structure of cells based on their relationships.

    Args:
        cell (Tuple[int, int, int, int]): Coordinates of the current cell as a tuple (x1, y1, x2, y2).
        rectangle_text_dict (dict): A dictionary mapping cell coordinates to their corresponding text.
        header_cell_dict (dict): A dictionary representing the relationships between header cells and their child cells.

    Returns:
        dict: A hierarchical structure of cells represented as a dictionary.
    """

    text = rectangle_text_dict.get(cell, None)
    # print(text)

    node = {
        text: []
    }

    for child_cell in header_cell_dict.get(cell, []):
        node[text].append(build_structure(child_cell, 
                                          rectangle_text_dict, 
                                          header_cell_dict))

    return node


def fill_structure(structure: dict, 
                   records_text: list) -> None:
    """
    Recursively fills the hierarchical structure of cells with data based on the relationships.

    Args:
        structure (dict): A hierarchical structure of cells represented as a dictionary.
        records_text (list): A list of text in current record line.

    Returns:
        None
    """

    if isinstance(structure, list):
        for item in structure:
            for key, value in item.items():
                if isinstance(value, list) and not value:
                    if records_text:
                        item[key] = records_text.pop(0)
                else:
                    fill_structure(value, records_text)
                    
    return structure


def split_records(record_cells: List[tuple], 
                  columns_for_records: int) -> List[List[tuple]]:
    """
    Splits a list of record cells into groups of cells, each representing a record.

    Args:
        record_cells (List[tuple]): A list of record cells, where each cell is represented as a tuple.
        columns_for_records (int): The number of columns for each record.

    Returns:
        List[List[tuple]]: A list of groups of cells, where each group represents a record.
    """

    for i in range(0, len(record_cells), columns_for_records):
        yield record_cells[i:i + columns_for_records]


def extract_record_text(records_list: List[Tuple[int, int, int, int]], 
                        text_dict: Dict[Tuple[int, int, int, int], List[str]])-> List[str]:
    """
    Extract text for each record cell.

    Args:
        records_list (List[Tuple[int, int, int, int]): List of record cell coordinates.
        text_dict (Dict[Tuple[int, int, int, int], Optional[str]): Dictionary of cell coordinates to text.

    Returns:
        List[Optional[str]]: List of text values for each record cell. None for missing cells.
    """
    records_list = sorted(records_list, key=lambda x: (x[0], x[1]))
    text_list = []
    for record_cell in records_list:
        if record_cell in text_dict:
            text_list.append(text_dict[record_cell])
        else:
            text_list.append(None)
    return text_list