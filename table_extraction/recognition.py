import os
import re
import cv2
import torch
import easyocr
import numpy as np
import multiprocessing as mp
from typing import List, Tuple
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor


def image_to_text_easyocr(image: np.ndarray, reader) -> str:
    """
    Perform text recognition on an image using EasyOCR.

    Args:
        image (np.ndarray): Input image as a NumPy array for text recognition.
        reader (easyocr.Reader): Initialized EasyOCR Reader instance.

    Returns:
        str: Recognized text from the input image.
    """

    # Perform text recognition on the input image
    result = reader.readtext(image, batch_size=16)
    # result = reader.recognize(image)

    # Extract and concatenate the recognized text from the result
    text = ''
    for detection in result:
        text += detection[1] + ' '
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


def filter_tables_by_classification(tables: List[np.ndarray],
                                    reader: easyocr.Reader) -> List[np.ndarray]:
    """
    Filters tables based on their classification.

    Args:
        tables (List[np.ndarray]): List of table images.
        reader (easyocr.Reader): Initialized EasyOCR Reader instance.

    Returns:
        List[np.ndarray]: List of filtered table images.
    """

    # Filtering tables based on information in their
    filtered_tables = []
    for table in tables:
        text = image_to_text_easyocr(table, reader)
        if classify_table(text):
            filtered_tables.append(table)
    return filtered_tables


def process_cell(image: np.ndarray, 
                 rectangle: Tuple[int, int, int, int], 
                 reader: easyocr.Reader) -> Tuple[Tuple[int, int, int, int], str]:
    """
    Process a cell in an image to extract text using EasyOCR.

    Args:
        image (np.ndarray): The image containing the cell.
        rectangle (Tuple[int, int, int, int]): Rectangle coordinates (x1, y1, x2, y2) of the cell.
        reader (easyocr.Reader): Initialized EasyOCR Reader instance.

    Returns:
        Tuple[Tuple[int, int, int, int], str]: A tuple containing the rectangle coordinates and recognized text.
    """

    x1, y1, x2, y2 = rectangle

    # Crop images at cell borders
    # margin = 5
    # cell_image = image[max(0, y1 - margin):min(image.shape[1], y2 + margin), max(0, x1 - margin):min(image.shape[0], x2 + margin)]
    cell_image = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

    # # Check if cell_image is not empty
    # if not cell_image.any():
    #     return (x1, y1, x2, y2), ""

    # Text recognition
    text = image_to_text_easyocr(cell_image, reader)

    # Store the recognized text in the cell_text dictionary with rectangle coordinates as the key
    return (x1, y1, x2, y2), text


def initial_reader(_) -> easyocr.Reader:
    """
    Initialize an EasyOCR Reader instance with the specified configuration.

    Returns:
        easyocr.Reader: Initialized EasyOCR Reader instance.
    """

    gpu_available = torch.cuda.is_available()
    return easyocr.Reader(
        ['en', 'ru'], 
        model_storage_directory='easy_ocr/model',
        user_network_directory='easy_ocr/user_network',
        gpu=gpu_available,
        verbose=False
    )


def remove_hyphenated_words(text: str) -> str:
    """
    Remove hyphenated words from the input text.

    Args:
        text (str): The input text containing hyphenated words.

    Returns:
        str: The modified text with hyphenated words removed.
    """
    
    pattern = r'(?<=[a-zA-Zа-яА-Я]) ?- ?(?=[a-zA-Zа-яА-Я])'
    result = re.sub(pattern, '', text)
    return result


def osr_detection(tables: List[np.ndarray], 
                  tables_cells: List[List[Tuple[int, int, int, int]]]) -> List[dict]:
    """
    Performs Optical Structure Recognition (OSR) on tables to extract cell text.

    Args:
        tables (List[np.ndarray]): List of table images.
        tables_cells (List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]): List of cells for each table image.
            Each rectangle is represented as a tuple of two points (top-left and bottom-right corners).

    Returns:
        List[dict]: List of dictionaries, where each dictionary represents cell text within the tables.
        The keys in the dictionary are rectangle coordinates, and the values are the recognized text within each cell.
    """

    # Initialize a list to store cell text for each table
    tables_cell_text = []

    # Check if a GPU is available
    gpu_available = torch.cuda.is_available()

    # Initialize the EasyOCR reader with language settings and model storage directories
    # num_workers = mp.cpu_count()
    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     readers = list(executor.map(initial_reader, range(num_workers)))

    # print(len(readers))
    
    reader = easyocr.Reader(['en', 'ru'], 
        model_storage_directory='easy_ocr/model',
        user_network_directory='easy_ocr/user_network',
        gpu=gpu_available,
        verbose=False)    

    # Loop through each table and its corresponding rectangles
    for num, image in enumerate(tables):
        cell_text = {}

        # images = [image for _ in range(len(tables_cells[num]))]

        # with ThreadPoolExecutor() as executor:
        #     cell_text = dict(executor.map(process_cell, [image for i in range(len(tables_cells[num]))], tables_cells[num]))

        # with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        #     cell_text = dict(executor.map(process_cell, [image for i in range(len(tables_cells[num]))], tables_cells[num], [reader for i in range(len(tables_cells[num]))]))

        # with ThreadPoolExecutor(max_workers=num_workers) as executor:
        #     cell_text = list(executor.map(process_cell, images, tables_cells[num], readers))

        # Iterate through cell within the current table
        for cell in tables_cells[num]:
            x1, y1, x2, y2 = cell

            # Crop images at cell borders
            # margin = 5
            # cell_image = image[max(0, y1 - margin):min(image.shape[1], y2 + margin), max(0, x1 - margin):min(image.shape[0], x2 + margin)]
            cell_image = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            
            # Text recognition
            text = ''
            if isinstance(cell_image, np.ndarray) and cell_image.size > 0:
                text = remove_hyphenated_words(image_to_text_easyocr(cell_image, reader))
            # print(text)

            # Store the recognized text in the cell_text dictionary with cell coordinates as the key
            cell_text[(x1, y1, x2, y2)] = text

        # Append the cell text dictionary for the current table to the result list
        tables_cell_text.append(cell_text)
        
    # Return the list of dictionaries, each representing cell text within the tables
    return tables_cell_text