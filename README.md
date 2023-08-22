## Table Detection and Recognition

This project aims to develop an end-to-end solution for table detection and recognition in documents and images. The primary goal is to build a system that can automatically detect tables within documents, extract the tabular data, and recognize the contents of the tables using optical character recognition (OCR) techniques.

## Technologies and Methods

The project utilizes a combination of computer vision, deep learning, and natural language processing techniques to achieve table detection and recognition. The following technologies and methods are employed:

1. **OpenCV**: OpenCV is used for image preprocessing and contour detection to identify potential table regions within documents.

2. **Mask R-CNN**: The Mask R-CNN architecture is employed for table segmentation and pixel-wise object detection. This deep learning model allows us to precisely delineate table boundaries.

3. **Tesseract OCR**: The Tesseract OCR engine is used for text recognition within the detected tables. It provides support for multiple languages and is well-suited for recognizing text in various formats.

4. **EasyOCR**: EasyOCR is another OCR library that is employed for recognizing text in the tables.

## Project Architecture

The project follows a modular architecture to enable easy extensibility and maintainability. The major components include:

1. **Table Detection**: This module focuses on using OpenCV and Mask R-CNN to detect table regions within documents and images.

2. **Table Recognition**: The table recognition module employs Tesseract OCR and EasyOCR to extract text from the detected tables.

3. **Data Post-Processing**: The extracted tabular data is processed with NumPy and Pandas to organize and present it in a structured format. such as yaml and json.

4. **User Interface**: A simple user interface is provided for interacting with the system, allowing users to upload documents and visualize the detected tables along with recognized text. For more customizable code interaction, recommend using TableExtraction.ipynb


# Step by Step Detection
To help with debugging and understanding the model, there are notebook 
([TableExtraction.ipynb](TableExtraction/TableExtraction.ipynb)), which provide many visualizations and allow you to step through the model to check the output at each point, change settings and see possible code errors that occur. Here are some examples:

## 1. File selection
First, select the PDF file from which you want to extract tabular information. Specify its address in the code:
```python
extractor.extract_from_file('PUT/YOUR/FILE/ADDRESS/HERE.pdf')
```
![](assets/page.png)

## 2. Finding a table on a page
Run the code. You will see a visualization of the table data recognition processes, the first of which will be finding the table on the page.

![](assets/mask.png)

## 3. Line detection
Further, in the area where the table was found at the previous step, lines are searched that bound the cells of the table.

![](assets/lines.png)

## 4.Table node detection
After finding the lines, you need to understand how the cells are located relative to each other. With the help of sequential processing, potential nodes are found, repetitions are removed, sorting occurs.

![](assets/nodes.png)

## 5. Cell division
Next, the cells are divided into those that store the names of the columns, and those that store the data.

![](assets/header.png)
![](assets/records.png)

## 6. Result
As a result of the algorithm, a file with the yaml extension will appear in the resilts folder, in which the structured data of each of the pages of the document will be stored

## Installation

<!-- Отдельно склонировать репозиторий mask rcnn
проблема с numpy
pip3 install -r requirements.txt
прописал в requirements версии
надо установить Build tools for visual studio 
microsoft visual c++ 14.0 is required get it with build tools for visual studio https //visualstudio.microsoft.com/downloads/ 
сначала установил питон python=3.7
отдельно mrcnn==0.2
pip3 install -r requirements.txt --user
https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick_on-windows -->

1. Clone this repository
    ```bash
   git clone https://github.com/DikovAlexandr/TableExtraction
   ```
2. Create a virtual environment (recommended) to isolate project dependencies
   ```bash
   conda create --name table_extraction python=3.7
   conda activate table_extraction
   ```
3. Install the required dependencies from the requirements.txt file, including the Mask R-CNN library
   ```bash
   pip3 install -r requirements.txt
   ```
   Please note that another repository "Mask-RCNN-TF2" should be cloned as a result

### Contributing

If you have any ideas, bug reports, or feature requests, feel free to open an issue or submit a pull request on the project's repository.

### License
The project is open-source and licensed under the Apache-2.0 license. You can find the details in the `LICENSE` file.
