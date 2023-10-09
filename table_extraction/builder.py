import cv2
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

def split_into_headers_and_records(rectangles: List[Tuple[Tuple[int, int], 
                Tuple[int, int]]]) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[Tuple[Tuple[int, int], Tuple[int, int]]], int, int]:
    """
    Splits a list of rectangles into headers and records based on the number of cells per row.

    Args:
        rectangles (List[Tuple[Tuple[int, int], Tuple[int, int]]]): A list of rectangles, where each rectangle is
        represented as a tuple of two points (top-left and bottom-right corners).

    Returns:
        Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], int, int]:
        - List of header rectangles, where each rectangle represents the header row.
        - List of record rectangles, where each rectangle represents a record row.
        - Number of cells per row (num_cells).
        - Total number of record rows (records).
    """
    cell_counts = {}  # Dictionary to store cell counts for each y

    for _, y1, _ in rectangles:
        if y1 in cell_counts:
            cell_counts[y1] += 1
        else:
            cell_counts[y1] = 1

    # Find the y where the number of cells changes
    sorted_cell_counts = sorted(cell_counts.items())
    # Number of cells of each record line
    num_cells = None
    records = 1

    for y1, count in sorted_cell_counts:
        if num_cells is None:
            num_cells = count
        elif count != num_cells:
            change_y = y1
            break
        else:
            records += 1

    record_cells = [((x1, y1), (x2, y2))
                    for (x1, y1), (x2, y2) in rectangles if y1 < change_y]
    header_cells = [((x1, y1), (x2, y2))
                    for (x1, y1), (x2, y2) in rectangles if y1 >= change_y]

    return header_cells, record_cells, num_cells, records

def visualize_headers_and_records_cells(image: np.ndarray, cells: List[Tuple[Tuple[int, int], Tuple[int, int]]], cell_type: str) -> None:
    """
    Visualizes header or record cells on an image.

    Args:
        image (np.ndarray): The image on which to visualize cells.
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
    copy_image = image.copy()
    copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)
    for cell in cells:
        (x1, y1), (x2, y2) = cell
        cv2.rectangle(copy_image, (x1, y1),
                        (x2, y2), color, 10)

    plt.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))
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

    for i, cell_1 in enumerate(cells):
        overlapping_cells = []
        x1_cell_1, y1_cell_1, x2_cell_1, y2_cell_1 = cell_1

        for j, cell_2 in enumerate(cells):
            if i != j:
                x1_cell_2, y1_cell_2, x2_cell_2, y2_cell_2 = cell_2
                if x1_cell_1 <= x1_cell_2 < x2_cell_2 <= x2_cell_1 and y1_cell_2 == y2_cell_1:
                    overlapping_cells.append(cell_2)
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
    node = {
        text: []
    }

    for child_cell in header_cell_dict.get(cell, []):
        node[text].append(build_structure(child_cell, 
                                          rectangle_text_dict, 
                                          header_cell_dict))

    return node

def fill_structure(structure: dict, 
                   data: list, 
                   rectangle_text_dict: dict) -> None:
    """
    Recursively fills the hierarchical structure of cells with data based on the relationships.

    Args:
        structure (dict): A hierarchical structure of cells represented as a dictionary.
        data (list): A list of cell data to fill into the structure.
        rectangle_text_dict (dict): A dictionary mapping cell coordinates to their corresponding text.

    Returns:
        None
    """
    data_copy = data.copy()
    if isinstance(structure, list):
        for item in structure:
            for key, value in item.items():
                if isinstance(value, list) and not value:
                    if data_copy:
                        item[key] = rectangle_text_dict.get(
                            data_copy.pop(0), '')
                else:
                    fill_structure(
                        value, data_copy, rectangle_text_dict)

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