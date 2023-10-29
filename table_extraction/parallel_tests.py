import os
import time
import torch
import random
import easyocr
import multiprocessing as mp
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore")

def process_image(image, rectangles, reader):
    torch.backends.quantized.engine = 'none'
    
    result = reader.readtext(image=image)

    # result = reader.recognize(image)

    # Extract and concatenate the recognized text from the result
    text = ''
    for detection in result:
        text += detection[1]

    return text

def init_reader(num_workers):
    gpu_available = torch.cuda.is_available()
    return easyocr.Reader(
        ['en', 'ru'], 
        model_storage_directory='easy_ocr/model',
        user_network_directory='easy_ocr/user_network',
        gpu=gpu_available,
        verbose=False
    )

if __name__ == "__main__":
    gpu_available = torch.cuda.is_available()
    input_folder = os.path.join(os.getcwd(), "input", "cells")
    images = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(".jpg")]
    rectangles = [(1, 2, 3, 4) for _ in range(len(images))]

    num_repeats = 1
    num_workers = mp.cpu_count()
    processing_times = {'ThreadPoolExecutor': [], 'Direct': []}

    reader = easyocr.Reader(['en', 'ru'], 
        model_storage_directory='easy_ocr/model',
        user_network_directory='easy_ocr/user_network',
        gpu=gpu_available,
        verbose=False)
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        readers = list(executor.map(init_reader, range(num_workers)))
    processing_time = time.time() - start_time

    print(processing_time)
    print(len(readers))
    print(num_workers)

    for method in ['ThreadPoolExecutor', 'Direct']:
        for i in range(num_repeats):
            results = []
            if method == 'ThreadPoolExecutor':
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    results += list(executor.map(process_image, images, rectangles, readers))
                end_time = time.time()
                processing_time = end_time - start_time
                print(processing_time)
                processing_times[method].append(processing_time)

                print(len(results))
                
            # if method == 'Direct':
            #     start_time = time.time()
            #     for image in images:
            #         results.append(process_image(image, rectangles, reader))
            #     end_time = time.time()
            #     processing_time = end_time - start_time
            #     print(processing_time)
            #     processing_times[method].append(processing_time)

    # print(processing_times)

    # for method, times in processing_times.items():
    #     average_time = sum(times) / len(times)
    #     print(f'Average time with {method}: {average_time} seconds')

    # print(len(results))

    # # Строим графики
    # plt.figure(figsize=(8, 6))
    # for method, times in processing_times.items():
    #     plt.plot(range(1, num_repeats + 1), times, label=method)
    # plt.title('Comparison of Processing Time')
    # plt.xlabel('Repeats')
    # plt.ylabel('Time (seconds)')
    # plt.legend()
    # plt.show()