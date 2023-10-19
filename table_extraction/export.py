import os
import io
import json
import yaml
import numpy as np
from typing import List, Tuple

from builder import (
    split_into_headers_and_records,
    split_into_headers_and_records_maskrcnn,
    visualize_headers_and_records_cells,
    create_cell_dict,
    build_structure,
    split_records,
    fill_structure
)


def make_yaml_file(tables: List[np.ndarray], 
                   tables_rectangles: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]], 
                   ocr_detected_text: List[dict]) -> List[dict]:
    """
    Creates a YAML representation of structured data extracted from tables.

    Args:
        tables (List[np.ndarray]): List of table images.
        tables_rectangles (List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]): List of rectangles for each table image.
            Each rectangle is represented as a tuple of two points (top-left and bottom-right corners).
        ocr_detected_text (List[dict]): List of dictionaries, where each dictionary represents cell text within the tables.
            The keys in the dictionary are rectangle coordinates, and the values are the recognized text within each cell.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary represents the structured data extracted from a table.
            The structure is built based on the detected headers and records in the tables.
    """
    results = []
    for num, image in enumerate(tables):
        structure = []
        if tables_rectangles[num]:
            # header_cells, record_cells, columns_for_records, records = split_into_headers_and_records_maskrcnn(image, tables_rectangles[num])
            header_cells, record_cells, columns_for_records, records = split_into_headers_and_records(tables_rectangles[num])

            if header_cells and record_cells:
                # visualize_headers_and_records_cells(image, header_cells, 'header')
                # visualize_headers_and_records_cells(image, record_cells, 'record')

                header_cell_dict = create_cell_dict(header_cells)
                rectangle_text_dict = ocr_detected_text[num]

                for cell in header_cells:
                    structure.append(build_structure(cell, 
                                                     rectangle_text_dict, 
                                                     header_cell_dict))
                    
                # 
                # print(structure)
                # 

                data_list = list(split_records(record_cells, 
                                               columns_for_records))
                
                # 
                # print(data_list)
                # 

                for i in range(records):
                    fill_structure(
                        structure, data_list[i], rectangle_text_dict)

        results.append(structure)
    return results

def save(results: List[dict], format: str, origin_file_path: str) -> None:
    """
    Saves the structured data extracted from tables to files in YAML or JSON format.

    Args:
        results (List[dict]): A list of dictionaries, where each dictionary represents structured data extracted from a table.
        format (str): The format in which to save the data ('yaml' or 'json').
        origin_file_path (str): The path to the original file from which the data was extracted.

    Returns:
        None
    """
    if results:
        if not os.path.exists("results"):
            os.makedirs("results")

        if format == 'yaml':
            for num, structure in enumerate(results):
                file_name = os.path.basename(origin_file_path)
                output_file = f"results/{os.path.splitext(file_name)[0]}_table_{num}.yaml"
                with open(output_file, 'w', encoding='utf-8') as yaml_file:
                    yaml.dump(structure, yaml_file,
                                default_flow_style=False, allow_unicode=True)
            return True

        if format == 'json':
            for num, structure in enumerate(results):
                file_name = os.path.basename(origin_file_path)
                output_file = f"results/{os.path.splitext(file_name)[0]}_table_{num}.json"
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    json.dump(structure, json_file,
                                ensure_ascii=False, indent=4)
            return True
    else: False
