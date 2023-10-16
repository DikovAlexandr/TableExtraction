from preprocessing import (grayzation, 
                           binarization,
                           pdf_file_to_array,
                           bytes_file_to_array,
                           visualize_images)
from detection import (get_nodes,
                       get_rectangles,
                       get_lines_Hough,
                       get_cells_maskrcnn, 
                       get_tables_maskrcnn,
                       visualize_table_images, 
                       visualize_rectangles)
from builder import split_into_headers_and_records_maskrcnn
from export import make_yaml_file, save
from recognition import osr_detection
import warnings

# Отключить все предупреждения (warnings)
warnings.filterwarnings("ignore")

def extract(file, filename):
    low_dpi = 50
    high_dpi = 500
    high_quality_images_array = bytes_file_to_array(file, high_dpi)
    low_quality_images_array = bytes_file_to_array(file, low_dpi)

    high_quality_gray_images_array = grayzation(high_quality_images_array)
    low_quality__gray_images_array = grayzation(low_quality_images_array)
    thresholded_images_array = binarization(high_quality_gray_images_array)
    tables, _ = get_tables_maskrcnn(low_quality_images_array,
                                 low_dpi,
                                 high_quality_images_array,
                                 high_dpi)
    # visualize_table_images(tables)

    tables_cells = get_cells_maskrcnn(tables)
    # visualize_rectangles(tables, tables_cells)

    # tables_lines = get_lines_Hough(tables)
    # tables_nodes = get_nodes(tables, tables_lines)
    # tables_cells = get_rectangles(tables, tables_nodes)
    # visualize_rectangles(tables, tables_cells)

    # split_into_headers_and_records_maskrcnn(tables[0], tables_cells[0])
    table_cell_text = osr_detection(tables, tables_cells)
    # table_cell_text = osr_detection_parallel(tables, tables_cells)
    results = make_yaml_file(tables, tables_cells, table_cell_text)
    save(results, 'yaml', filename)