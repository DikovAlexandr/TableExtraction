import os
import time
import torch
import random
import easyocr
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def process_image(image, reader, batch_size, workers):
    result = reader.readtext(image=image, batch_size=batch_size, workers=workers)

    # result = reader.recognize(image)

    # Extract and concatenate the recognized text from the result
    text = ''
    for detection in result:
        text += detection[1]

    return image, text

if __name__ == "__main__":
    gpu_available = torch.cuda.is_available()
    input_folder = os.path.join(os.getcwd(), "input", "cache")
    images = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(".jpg")]

    best_time = float('inf')
    best_params = (1, 0)

    for batch_size in range(1, 9):
        for workers in range(0, os.cpu_count() + 1):
            start_time = time.time()
            results = []
            reader = easyocr.Reader(['en', 'ru'], 
                model_storage_directory='easy_ocr/model',
                user_network_directory='easy_ocr/user_network',
                gpu=gpu_available,
                verbose=False)

            for image in images:
                results.append(process_image(image, reader, batch_size, workers))

            end_time = time.time()
            processing_time = end_time - start_time

            if processing_time < best_time:
                best_time = processing_time
                best_params = (batch_size, workers)

            print(f"Parameters: batch_size={batch_size}, workers={workers}, Time: {processing_time} seconds")

    print(f"Best Parameters: batch_size={best_params[0]}, workers={best_params[1]}, Best Time: {best_time} seconds")