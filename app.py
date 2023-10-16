from flask import (Flask, 
                   request,
                   send_file,  
                   render_template,                    
                   send_from_directory)
import sys
import os

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'table_extraction'))
sys.path.insert(0, module_dir)

import extractor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            filename = uploaded_file.filename
            result_file = extractor.extract(uploaded_file.read(), filename)
            if result_file:
                return send_file(result_file, as_attachment=True)
            else:
                return "Error of processing PDF"
    return render_template('upload.html')

@app.route('/downloads')
def download_results():
    files = os.listdir(r"table_extraction\results")  # Получить список файлов в директории
    return render_template('downloads.html', files=files)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(r"table_extraction\results", filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
