import os
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
        reader : Reader for text recognition (EasyOCR)
    Returns:
        str: Recognized text from the input image.
    """
    # Perform text recognition on the input image
    result = reader.readtext(image, batch_size=16)
    # result = reader.recognize(image)

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

def process_cell(image, rectangle, reader):
    x1, y1, x2, y2 = rectangle

    # Crop images at cell borders
    # margin = 5
    # cell_image = image[max(0, y1 - margin):min(image.shape[1], y2 + margin), max(0, x1 - margin):min(image.shape[0], x2 + margin)]
    cell_image = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

    # Check if cell_image is not empty
    if not cell_image.any():
        return (x1, y1, x2, y2), ""

    # Text recognition
    text = image_to_text_easyocr(cell_image, reader)
    # print(text)

    # Store the recognized text in the cell_text dictionary with rectangle coordinates as the key
    return (x1, y1, x2, y2), text

def initial_reader(_):
    gpu_available = torch.cuda.is_available()
    return easyocr.Reader(
        ['en', 'ru'], 
        model_storage_directory='easy_ocr/model',
        user_network_directory='easy_ocr/user_network',
        gpu=gpu_available,
        verbose=False
    )

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

    # Check if a GPU is available
    gpu_available = torch.cuda.is_available()

    # Initialize the EasyOCR reader with language settings and model storage directories
    num_workers = mp.cpu_count()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        readers = list(executor.map(initial_reader, range(num_workers)))
    # reader = easyocr.Reader(['en', 'ru'], 
    #     model_storage_directory='easy_ocr/model',
    #     user_network_directory='easy_ocr/user_network',
    #     gpu=gpu_available,
    #     verbose=False)

    # Loop through each table and its corresponding rectangles
    for num, image in enumerate(tables):
        cell_text = {}

        # with ThreadPoolExecutor() as executor:
        #     cell_text = dict(executor.map(process_cell, [image for i in range(len(tables_rectangles[num]))], tables_rectangles[num]))

        # with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        #     cell_text = dict(executor.map(process_cell, [image for i in range(len(tables_rectangles[num]))], tables_rectangles[num], [reader for i in range(len(tables_rectangles[num]))]))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            cell_text = dict(executor.map(process_cell, tables_rectangles[num], readers))

        # # Iterate through rectangles within the current table
        # for rectangle in tables_rectangles[num]:
        #     x1, y1, x2, y2 = rectangle

        #     # Crop images at cell borders
        #     # margin = 5
        #     # cell_image = image[max(0, y1 - margin):min(image.shape[1], y2 + margin), max(0, x1 - margin):min(image.shape[0], x2 + margin)]
        #     cell_image = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        #     image_filename = f"cell_{abs(x1-x2)}_{abs(y1-y2)}.jpg"
        #     image_path = os.path.join(os.getcwd(), image_filename)
        #     cv2.imwrite(image_path, cell_image)

        #     # Text recognition
        #     text = image_to_text_easyocr(cell_image, reader)
        #     # print(text)

        #     # Store the recognized text in the cell_text dictionary with rectangle coordinates as the key
        #     cell_text[(x1, y1, x2, y2)] = text

        # Append the cell text dictionary for the current table to the result list
        tables_cell_text.append(cell_text)
        
    # Return the list of dictionaries, each representing cell text within the tables
    return tables_cell_text