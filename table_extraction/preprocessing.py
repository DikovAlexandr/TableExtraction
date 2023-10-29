import os
import cv2
import platform
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from pdf2image import convert_from_path, convert_from_bytes 


def bytes_file_to_array(pdf_bytes: bytes, dpi: int = 300) -> List[np.ndarray]:
    """
    Converts a PDF file to a list of NumPy arrays representing images.

    Args:
        pdf_bytes (bytes): The PDF content in bytes.
        dpi (int, optional): Dots per inch for image conversion. Defaults to 300.

    Returns:
        List[np.ndarray]: A list of NumPy arrays representing images.
    """
    system = platform.system()

    poppler_path = os.path.join(os.environ.get(
        'PROGRAMFILES', 'C:\\Program Files'), 'poppler-23.07.0', 'Library', 'bin')
    
    images = convert_from_bytes(    
        pdf_bytes, dpi=dpi, poppler_path=poppler_path)
    return [np.array(image) for image in images]


def pdf_file_to_array(file_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Converts a PDF file to a list of NumPy arrays representing images.

    Args:
        file_path (str): The path to the PDF file.
        dpi (int, optional): Dots per inch for image conversion. Defaults to 300.

    Returns:
        List[np.ndarray]: A list of NumPy arrays representing images.
    """
    poppler_path = os.path.join(os.environ.get(
        'PROGRAMFILES', 'C:\\Program Files'), 'poppler-23.07.0', 'Library', 'bin')
    images = convert_from_path(
        file_path, dpi=dpi, poppler_path=poppler_path)
    return [np.array(image) for image in images]


def image_file_to_array(file_path: str) -> List[np.ndarray]:
    """
    Reads an image file and converts it to a list of NumPy arrays.

    Args:
        file_path (str): The path to the image file.

    Returns:
        List[np.ndarray]: A list of NumPy arrays representing the images.
    """
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = np.array(image_rgb)
    return [image_array]


def visualize_images(images: List[np.ndarray]) -> None:
    """
    Visualizes a list of images.

    Args:
        images (List[np.ndarray]): A list of NumPy arrays representing images.

    Returns:
        None
    """
    for image in images:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


def grayzation(images_array: List[np.ndarray]) -> List[np.ndarray]:
    """
    Converts a list of color images to grayscale.

    Args:
        images_array (List[np.ndarray]): A list of NumPy arrays representing color images.

    Returns:
        List[np.ndarray]: A list of NumPy arrays representing grayscale images.
    """
    gray_images = []
    for image in images_array:
        gray_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return gray_images


def binarization(images_array: List[np.ndarray]) -> List[np.ndarray]:
    """
    Converts a list of grayscale images to binary images.

    Args:
        images_array (List[np.ndarray]): A list of NumPy arrays representing grayscale images.

    Returns:
        List[np.ndarray]: A list of NumPy arrays representing binary images.
    """
    threshold_images = []
    for gray_image in images_array:
        _, threshold_image = cv2.threshold(
            gray_image, 200, 255, cv2.THRESH_BINARY)
        threshold_images.append(threshold_image)
    return threshold_images
