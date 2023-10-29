## Table Detection and Recognition

This project aims to develop an end-to-end solution for table detection and recognition in documents and images. The primary goal is to build a system that can automatically detect tables within documents, extract the tabular data and recognize the contents of the tables using optical character recognition (OCR) techniques.

## Technologies and Methods

The project utilizes a combination of computer vision, deep learning and natural language processing techniques to achieve table detection and recognition. The following technologies and methods are employed:

1. **OpenCV**: OpenCV is used for image preprocessing and contour detection to identify potential table regions within documents.

2. **Mask R-CNN**: The Mask R-CNN architecture is employed for table segmentation and pixel-wise object detection. This deep learning model allows us to precisely delineate table boundaries.

3. **EasyOCR**: The EasyOCR engine is used for text recognition within the detected tables. It provides support for multiple languages and is well-suited for recognizing text in various formats.

## Project Architecture

The project follows a modular architecture to enable easy extensibility and maintainability. The major components include:

1. **Table Detection**: This module focuses on using OpenCV and Mask R-CNN to detect table regions within documents and images.

2. **Table Recognition**: The table recognition module employs EasyOCR to extract text from the detected tables.

3. **Data Post-Processing**: The extracted tabular data to organize and present it in a structured format, such as yaml or json.

4. **User web-interface**: A simple user web-interface is provided for interacting with the system, allowing users to upload documents and download detected tables along with recognized text. But TableExtraction.ipynb is recommend for more customizable code interaction.

# Step by Step Detection
To help with debugging and understanding the model, there is notebook [TableExtraction.ipynb](TableExtraction/TableExtraction.ipynb), which provide many visualizations and allow you to step through the model to check the output at each point, change settings and see possible code errors that occur. Here are some examples:

## 1. File selection
First, select the PDF file which you want to extract tabular information from. Specify its address in the code:
```python
extractor.extract('PUT/YOUR/FILE/ADDRESS/HERE.pdf')
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

## 5.Cells detection
Once we've established the table's structure, the next step is to detect individual cells within it.
 - **Mask R-CNN**: We can use Mask R-CNN to directly recognize and segment individual cells within the table.
 - **Empirically**: Alternatively, we can rely on empirical rules based on the table's structure and layout to infer cell positions.

![](assets/cells.png)

## 6. Separating headers and records
Next, the cells are divided into those that store the names of the columns and data-storing cells.
 - **Mask R-CNN**: We can employ Mask R-CNN to recognize header and data cells.
 - **Empirically**: Based on prior knowledge about table formatting, we can use heuristics and logic to distinguish header cells from data cells.

![](assets/header.png)
![](assets/records.png)

## 7. Result
As a result of the algorithm, a file with the yaml extension will appear in the results folder, in which the structured data of each of the pages of the document will be stored

```yaml
- 'Наимснованис стали ': 'С235 '
- 'Толщина полки, ям ': 'От 4 до 20 включ. Св: 20 40 '
- 'Предел текучести Н/мм? кгс /мм? 6x ': '235(24) 225(23) '
- 'нс менсс ': '360(37) 360(37) '
- 'Временное сопротивление 0 н" Н/мм? кгс/мм? ': '26 25 '
- 'Относительное удлиненис cс 55 ': 'd = a d = 2a '
- 'Изгиб до параллсльности сторон (q толщина образца; d_диамстр оправки) ': ''
- 'при тсмпсратурс , C ':
  - '_20 ': ''
  - '_40 ': ''
  - '_70 ': ''
- '_20 ': []
- 'не менее ': []
- 'Ударная вязкость KCU, Дж/см? KгС м/см? ':
  - 'послс механичсского старсНИя ': []
- '_40 ': []
- '_70 ': []
- 'послс механичсского старсНИя ': []
```

## Examples

Here are five examples to illustrate how to use our software:

1. **Example 1:** Extracting tables from an image:

   <img src="assets/example_1.jpg" width="500" height="250">

   ```yaml
      - 'сТали Марка ':
      - 'Ст0 ':
         - 'Ст2кп ': []
      - 'СтЗкп ': []
      - 'ЕтЗпс,  СтЗсп ':
      - '~тЗГпс ':
         - 'ЕтЗГсп ':
            - 'СтАпс,  СтАсп Етэпс Стэсп т?гпа ': []
            - 'СтАпс,   СтАсп ': []
      - 'СтАпс,   СтАсп ': []
      - '1 330-430(34-44) ':
      - '360-460(37-47) ': []
      - '360-460(37-47) ': []
      - 'Неменее 300 (31) | ':
      - '320-410(33-42) ': []
      - '410-530(42-54) ':
      - '490-630(50-64) ': []
      - '370-490(38-50) ':
      - '390-570(40-58) ': []
      - 'Временное сопротивление Н/мм'' (кгс/мм Ов? ': []
      - '320-410(33-42) ': []
      - '370-480(38-49) ':
      - '370-490(38-50) ':
         - '390-570(40-58) ': []
      - '390-570(40-58) ': []
      - '490-630(50-64) ': []
   ```

2. **Example 2:** Extracting tables from an image:

   <img src="assets/example_2.jpg" width="400" height="150">

   ```yaml
      - 'ОппнячЕниг сталн ':
      - 'Сm ': 'Cть '
      - 'Сm ': 'U,Jg 7 0,49 '
      - 'Cт5 ': '0,49 {,18 '
      - 'Cт] ':
      - 'Cтz ': []
      - 'Cтz ': []
      - 'Ст ': []
      - 'Cт7 ':
      - 'Ст ': []
   ```

3. **Example 3:** Extracting tables from an image:

   <img src="assets/example_3.jpg" width="500" height="150">

   ```yaml
      - '; Марка стали ':
      - 'Новое обозначение ': 'a 0,40,6 0,25_ 0,35 0,8_1,1 0,8_1,1 0,25  0,35 0,50,7 0,35_ '
      - 'Старое обозначение ': []
      - 'Новое обозначение I2MX 12Х1МФ 2ОХ1М]ФIТР 2ОХ1МОФ]БР 25Х1МФ 18ХЗМВ 2ОХЗМВФ ': []
      - 'I2MX 12Х1МФ 20Х1М]Ф]ТР 2ОХ1М]Ф]БР ': []
      - 'Новое обозначение ': []
      - 'Старое обозначение ': []
      - 'чение 12ХМФ ЭП182 20ХМФБ ЭП44 ЭИ10 ЭИ578 ЭИ415, ЭИ579 ': []
      - '0,09_ 0,16 0,10_ 0,15 0,17- 0,24 0,18 0,25 0,22  0,29 0,15_ 0,20 0, 0,23 15_ ': []
      - 'Углерод ': []
      - '0,17 0,37 0,17 0,37 Hе более 0,37 He более 0,37 0,17 0,37 0,17 0,37 0,17- 0,37 ': []
      - 'ниЙ Крем- ': []
      - '0,40,7 0,40,7 Hе более 0,5 0,5-0,8 0,4_0,7 0,25_ 0,50 0,25_ 0,50 ': []
      - 'Марганец ':
      - '0,40,7 0,40,7 Hе более 0,5 0,5-0,8 0,4_0,7 0,25_ 0,50 0,25_ 0,50 ': []
      - '0,40,7 0,9-1,2 0,9-1,4 1,0-1,5 1,5-1,8 2,5-3,0 2,8-3,3 Хром ': []
      - 'Хром ': []
      - '1 Ста He более 0,30 Hе более 0,30 Hе более 0,30 Hе более 0,30 Hе более 0,30 Hе более 0,30 Hе более 0,30 ': []
      - 'Никель ':
      - '1 Ста He более 0,30 Hе более 0,30 Hе более 0,30 Hе более 0,30 Hе более 0,30 Hе более 0,30 Hе более 0,30 ': []
   ```

4. **Example 4:** Extracting tables from an image:

   <img src="assets/example_4.jpg" width="700" height="300">

   ```yaml
      - 'Марка стали ':
      - 'Старос обозначение ': '670-690 700750 680720 '
      - 'I2MX 12Х1МФ 2ОХ1М1ФIТР ': '21 45 59 21 55 98( 15 50 59 '
      - 'Новое обозначение ':
      - 'I2MX 12Х1МФ 2ОХ1М1ФIТР ': '45 55 50 '
      - 'Старос обозначение ': '59(6) 98(10 59(6) '
      - '12ХМФ ЭП182 ': []
      - 'Нормализация 910930 Нормализация 960-980 Закалка 970990 ': []
      - 'Температуpa нагрева , ':
      - 'Нормализация 910930 Нормализация 960-980 Закалка 970990 ': []
      - 'Закалка; нормализация, ОТЖИГ ':
      - 'Температуpa нагрева , ':
         - 'Нормализация 910930 Нормализация 960-980 Закалка 970990 ': []
      - 'Среда охлаждения ': []
      - 'Рекомендуемый режим термической обработки ':
      - 'Отпуск или старение ': []
      - 'Среда охлаждения ': []
      - 'Воздух Воздух Масло ': []
      - 'Отпуск или старение ': []
      - 'Температура нагрева, ': []

      - 'Способ выплавки стали ': 'Открытая выплавка Метод  электрошлаковоГО переплава '
      - 'Способ выплавки стали ': '2 '
      - 'Центральная пористость ': '2 '
      - 'Макроструктура B баллах, не ':
      - 'Точечная неоднородность ': '2 '
      - 'Точечная неоднородность ': '2 '
      - 'акроструктура в баллах; н€ более ':
      - 'Ликвационный квадрат ': []
      - 'Общая пятнистая ликвация ': []
      - 'Ликвационный квадрат ': []
      - 'Общая пятнистая ликвация ': []
   ```

5. **Example 5:** Extracting tables from an image:

   <img src="assets/example_5.jpg" width="500" height="100">

   ```yaml
   - 'Способ выплавки стали ': 'Открытая выплавка Метод  электрошлаковоГО переплава '
   - 'Способ выплавки стали ': '2 '
   - 'Центральная пористость ': '2 '
   - 'Макроструктура B баллах, не ':
   - 'Точечная неоднородность ': '2 '
   - 'Точечная неоднородность ': '2 '
   - 'акроструктура в баллах; н€ более ':
   - 'Ликвационный квадрат ': []
   - 'Общая пятнистая ликвация ': []
   - 'Ликвационный квадрат ': []
   - 'Общая пятнистая ликвация ': []
   ```

## Disclaimer

Please note that the algorithm provided here may not always produce perfect results and may require further refinements. The accuracy of table extraction can be influenced by various factors, including the quality and complexity of input documents. I recommend using this algorithm as a starting point and fine-tuning it based on your specific requirements and datasets. Contributions and improvements to enhance the algorithm's performance are welcome.

## Installation

Before using this code, make sure you have the following prerequisites installed:

- Microsoft Visual C++ 14.0: You can download it from [here](https://visualstudio.microsoft.com/downloads/).

- ImageMagick: Install ImageMagick by following the guidelines provided [here](https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick_on-windows).

- Poppler: Ensure you have Poppler version 23.07.0 or newer installed.You can download it from [here](https://github.com/oschwartz10612/poppler-windows/releases/) (only Windows) or from [here](https://poppler.freedesktop.org/). You can either place it in **C:\Program Files** on Windows, **/usr/bin** on Linux or modify the path in the [preprocessing.py](table_extraction/preprocessing.py) file as needed.

Now you can get started, follow these steps:

1. Clone this repository
   ```bash
   git clone https://github.com/DikovAlexandr/TableExtraction
   ```
2. Create a virtual environment (recommended) to isolate project dependencies
   ```bash
   # On Unix/Linux
   python3 -m venv table_extraction_venv

   # On Windows
   python -m venv table_extraction_venv
   ```
   ```bash
   # On Unix/Linux
   source table_extraction_venv/bin/activate

   # On Windows
   table_extraction_venv\Scripts\activate
   ```
3. Install the required dependencies from the requirements.txt file and setup.py:
   ```bash
   pip3 install -r requirements.txt
   ```
4. Downloading Pre-trained Model Weights:

   Before running the code, you have the option to download pre-trained model weights. You can choose between two versions: full precision (original) weights and quantized weights for memory efficiency.

   - Full Precision Weights:

   ```bach
   wget https://www.dropbox.com/scl/fi/loj552c7bllqsc5kwwffi/detect_table_plot.pth?rlkey=py4nqzrylmbtburmwf5m8814b&dl=0 -O detect_table_plot.pth
   ```

   ```bach
   wget https://www.dropbox.com/scl/fi/weijpwjbo19ucjmkbsh4b/best_cell_detection.pth?rlkey=6vuebzlez79uq53p4o3mc5voe&dl=0 -O best_cell_detection.pth
   ```

   - Quantized Weights:

   ```bach
   wget https://www.dropbox.com/scl/fi/g4sw2hf0flzf7vr3k2d64/detect_table_plot_quantized_model.pth?rlkey=ke37t2cabejkitsgmzrux883p&dl=0 -O detect_table_plot_quantized_model.pth
   ```

   ```bach
   wget https://www.dropbox.com/scl/fi/vd9q1j3mxhja2kloyq8bm/best_cell_detection_quantized_model.pth?rlkey=qwnowcjv20mvuuw2jf2mk0xar&dl=0 -O best_cell_detection_quantized_model.pth
   ```

### Contributing

If you have any ideas, bug reports, or feature requests, feel free to open an issue or submit a pull request on the project's repository.

### License
The project is open-source and licensed under the Apache-2.0 license. You can find the details in the `LICENSE` file.
