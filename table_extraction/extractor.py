from preprocessing import (grayzation, 
                           binarization,
                           pdf_file_to_array,
                           bytes_file_to_array,
                           visualize_images)

from detection import (get_nodes,
                       get_cells,
                       visualize_cells,
                       get_lines_Hough,
                       get_cells_maskrcnn, 
                       get_tables_maskrcnn,
                       visualize_table_images)

from recognition import osr_detection
from export import make_serialized_structure, save
from builder import split_into_headers_and_records_maskrcnn

import warnings
warnings.filterwarnings("ignore")


def extract(file, filename):
    low_dpi = 50
    high_dpi = 500

    try:
        high_quality_images_array = bytes_file_to_array(file, high_dpi)
        low_quality_images_array = bytes_file_to_array(file, low_dpi)

        high_quality_gray_images_array = grayzation(high_quality_images_array)
        low_quality__gray_images_array = grayzation(low_quality_images_array)
        thresholded_images_array = binarization(high_quality_gray_images_array)
    except Exception as e:
        print(f"File reading error: {str(e)}")
        return False  

    try:          
        tables, _ = get_tables_maskrcnn(low_quality_images_array,
                                    low_dpi,
                                    high_quality_images_array,
                                    high_dpi)
        # visualize_table_images(tables)
    except Exception as e:
        print(f"Error in table detection: {str(e)}")
        return False

    try:
        tables_cells = get_cells_maskrcnn(tables)
        # visualize_rectangles(tables, tables_cells)
    except Exception as e:
            print(f"Error in cells detection: {str(e)}")
            return False

    # tables_lines = get_lines_Hough(tables)
    # tables_nodes = get_nodes(tables, tables_lines)
    # tables_cells = get_cells(tables, tables_nodes)
    # visualize_cells(tables, tables_cells)

    # split_into_headers_and_records_maskrcnn(tables[0], tables_cells[0])

    try:
        table_cell_text = osr_detection(tables, tables_cells)
    except Exception as e:
            print(f"Error in cell text detection: {str(e)}")
            return False

    results = make_serialized_structure(tables, tables_cells, table_cell_text)
    result_path = save(results, 'yaml', filename)    
    return result_path