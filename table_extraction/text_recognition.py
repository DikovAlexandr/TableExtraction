import os
import cv2
import torch
import easyocr
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

def image_to_text_easyocr(image: np.ndarray) -> str:
    """
    Perform text recognition on an image using EasyOCR.

    Args:
        image (np.ndarray): Input image as a NumPy array for text recognition.

    Returns:
        str: Recognized text from the input image.
    """
    # Check if a GPU is available
    gpu_available = torch.cuda.is_available()

     # Initialize the EasyOCR reader with language settings and model storage directories
    reader = easyocr.Reader(['en', 'ru'], 
        model_storage_directory='easyocr/model',
        user_network_directory='easyocr/user_network',
        gpu=gpu_available )
    
    # Perform text recognition on the input image
    result = reader.readtext(image)

    # Extract and concatenate the recognized text from the result
    text = ''
    for detection in result:
        text += detection[1]
    return text

def classify_table(table_text: str) -> bool:
    """
    Classifies a table based on keywords found in its text content.

    Args:
        table_text (str): Text content extracted from a table.

    Returns:
        bool: True if the table is classified as valid, False otherwise.
    """
    keywords = ["марка", "стали", "временное", "сопротивление",
                        "предел", "текучести", "относительное", "удлинение"]
    error_threshold = 0.2
    keyword_importance = 0.8

    keywords_found = []

    def partial_compare(word, keyword):
        max_length = max(len(word), len(keyword))
        errors = sum(1 for w, k in zip(word, keyword) if w != k)
        error_ratio = errors / max_length
        return error_ratio <= error_threshold

    for keyword in keywords:
        found = any(partial_compare(keyword, word.lower())
                    for word in table_text.split())
        if found:
            keywords_found.append(keyword)

    total_keywords = len(keywords)
    found_keywords = len(keywords_found)
    confidence = keyword_importance * \
        (found_keywords / total_keywords)

    if confidence >= error_threshold:
        return True
    else:
        return False

def filter_tables_by_classification(tables: List[np.ndarray]) -> List[np.ndarray]:
    """
    Filters tables based on their classification.

    Args:
        tables (List[np.ndarray]): List of table images.

    Returns:
        List[np.ndarray]: List of filtered table images.
    """
    # Filtering tables based on information in their
    filtered_tables = []
    for table in tables:
        text = image_to_text_easyocr(table)
        if classify_table(text):
            filtered_tables.append(table)
    return filtered_tables


def osr_detection(tables: List[np.ndarray], tables_rectangles: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]) -> List[dict]:
    """
    Performs Optical Structure Recognition (OSR) on tables to extract cell text.

    Args:
        tables (List[np.ndarray]): List of table images.
        tables_rectangles (List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]): List of rectangles for each table image.
            Each rectangle is represented as a tuple of two points (top-left and bottom-right corners).

    Returns:
        List[dict]: List of dictionaries, where each dictionary represents cell text within the tables.
        The keys in the dictionary are rectangle coordinates, and the values are the recognized text within each cell.
    """
    # Initialize a list to store cell text for each table
    tables_cell_text = []

    # Loop through each table and its corresponding rectangles
    for num, image in enumerate(tables):
        cell_text = {}

        # Iterate through rectangles within the current table
        for rectangle in tables_rectangles[num]:
            x1, y1, x2, y2 = rectangle

            # Crop images at cell borders
            margin = 5
            cell_image = image[max(0, y1 - margin):min(image.shape[0], y2 + margin), max(0, x1 - margin):min(image.shape[1], x2 + margin)]
            
            # Text recognition
            text = image_to_text_easyocr(cell_image)
            print(text)

            # Store the recognized text in the cell_text dictionary with rectangle coordinates as the key
            cell_text[(x1, y1), (x2, y2)] = text

        # Append the cell text dictionary for the current table to the result list
        tables_cell_text.append(cell_text)
        
    # Return the list of dictionaries, each representing cell text within the tables
    return tables_cell_text

def split_into_headers_and_records(self, rectangles):
    max_y = max(y1 for (x1, y1), (x2, y2) in rectangles)

    prev_y = None
    cell_count = 0
    cell_counts = {}  # Dictionary to store cell counts for each y

    cell_counts = {}
    for (x1, y1), _ in rectangles:
        if y1 in cell_counts:
            cell_counts[y1] += 1
        else:
            cell_counts[y1] = 1

    # Find the y where the number of cells changes
    sorted_cell_counts = sorted(cell_counts.items())
    previous_value = None
    records = 1

    for y1, count in sorted_cell_counts:
        if previous_value is None:
            previous_value = count
        elif count != previous_value:
            change_y = y1
            break
        else:
            records += 1

    record_cells = [((x1, y1), (x2, y2))
                    for (x1, y1), (x2, y2) in rectangles if y1 < change_y]
    header_cells = [((x1, y1), (x2, y2))
                    for (x1, y1), (x2, y2) in rectangles if y1 >= change_y]

    return header_cells, record_cells, previous_value, records

def show_headers_and_records_cells(self, image, cells, type):
    if type == 'header':
        color = (120, 180, 0)
    elif type == 'record':
        color = (0, 180, 180)
    copy_image = image.copy()
    copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)
    height, width, _ = copy_image.shape
    for cell in cells:
        (x1, y1), (x2, y2) = cell
        cv2.rectangle(copy_image, (x1, height - y1),
                        (x2, height - y2), color, 10)

    plt.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def create_cell_dict(self, image, cells):
    # Create a dictionary in which the keys will be the coordinates of the current cell,
    # and the values will be the coordinates of the cells immediately below the current one.
    rectangle_dict = {}

    for i, cell1 in enumerate(cells):
        overlapping_cells = []
        (x1_cell1, y1_cell1), (x2_cell1, y2_cell1) = cell1

        for j, cell2 in enumerate(cells):
            if i != j:
                (x1_cell2, y1_cell2), (x2_cell2, y2_cell2) = cell2
                if x1_cell1 <= x1_cell2 < x2_cell2 <= x2_cell1 and y1_cell2 == y2_cell1:
                    overlapping_cells.append(cell2)

        # self.show_cells_relationship(image, cell1, overlapping_cells)
        rectangle_dict[cell1] = overlapping_cells

    return rectangle_dict

def show_cells_relationship(self, image, main_cell, overlapping_cells):
    copy_image = image.copy()
    copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)
    height, width, _ = copy_image.shape

    (x1, y1), (x2, y2) = main_cell
    cv2.rectangle(copy_image, (x1, height - y1),
                    (x2, height - y2), (120, 180, 0), 10)

    for cell in overlapping_cells:
        (x1, y1), (x2, y2) = cell
        cv2.rectangle(copy_image, (x1, height - y1),
                        (x2, height - y2), (0, 180, 180), 10)

    plt.imshow(cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def build_structure(self, cell, rectangle_text_dict, header_cell_dict):
    text = rectangle_text_dict.get(cell, None)
    node = {
        text: []
    }

    for child_cell in header_cell_dict.get(cell, []):
        node[text].append(self.build_structure(
            child_cell, rectangle_text_dict, header_cell_dict))

    return node

def fill_structure(self, structure, data, rectangle_text_dict):
    data_copy = data.copy()
    if isinstance(structure, list):
        for item in structure:
            for key, value in item.items():
                if isinstance(value, list) and not value:
                    if data_copy:
                        item[key] = rectangle_text_dict.get(
                            data_copy.pop(0), '')
                else:
                    self.fill_structure(
                        value, data_copy, rectangle_text_dict)

def split_records(self, record_cells, columns_for_records):
    for i in range(0, len(record_cells), columns_for_records):
        yield record_cells[i:i + columns_for_records]

def make_yaml_file(self):
    results = []
    for num, image in enumerate(self.tables):
        structure = []
        if self.tables_rectangles[num]:
            header_cells, record_cells, columns_for_records, records = self.split_into_headers_and_records(
                self.tables_rectangles[num])

            if header_cells and record_cells:
                self.show_headers_and_records_cells(image, header_cells, type='header')
                self.show_headers_and_records_cells(image, record_cells, type='record')

                header_cell_dict = self.create_cell_dict(
                    image, header_cells)
                rectangle_text_dict = self.tables_cell_text[num]

                for cell in header_cells:
                    structure.append(self.build_structure(
                        cell, rectangle_text_dict, header_cell_dict))

                data_list = list(self.split_records(
                    record_cells, columns_for_records))

                for i in range(records):
                    self.fill_structure(
                        structure, data_list[i], rectangle_text_dict)

                # structure = yaml.dump(structure, allow_unicode=True, indent=4)

        results.append(structure)
    return results

def save(self, format):
    if self.results:
        if not os.path.exists("results"):
            os.makedirs("results")

        if format == 'yaml':
            for num, structure in enumerate(self.results):
                file_name = os.path.basename(self.file_path)
                output_file = f"results/{os.path.splitext(file_name)[0]}_table_{num}.yaml"
                with open(output_file, 'w') as yaml_file:
                    yaml.dump(structure, yaml_file,
                                default_flow_style=False, allow_unicode=True)

        if format == 'json':
            for num, structure in enumerate(self.results):
                file_name = os.path.basename(self.file_path)
                output_file = f"results/{os.path.splitext(file_name)[0]}_table_{num}.json"
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    json.dump(structure, json_file,
                                ensure_ascii=False, indent=4)
