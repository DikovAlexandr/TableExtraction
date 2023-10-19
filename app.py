from flask import (Flask,
                   url_for,
                   request,
                   redirect,
                   send_file,  
                   render_template,                    
                   send_from_directory)
import sys
import os
import io

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'table_extraction'))
sys.path.insert(0, module_dir)

import extractor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            try:            
                filename = uploaded_file.filename
                results = extractor.extract(uploaded_file.read(), filename)
                if results:
                    return redirect(url_for('download_results'))
                else:
                    return "Error of processing PDF"
            except Exception as e:
                return "Error: " + str(e)
    return render_template('upload.html')

# @app.route('/downloads')
# def download_results():
#     files = os.listdir(r"table_extraction\results")
#     return render_template('downloads.html', files=files)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(r"table_extraction\results", filename, as_attachment=True)

@app.route('/download_results', methods=['GET'])
def download_results():
    result_files = os.listdir(os.path.join(os.getcwd(), "results"))
    print(os.path.join(os.getcwd(), "results"))
    print(result_files)
    return render_template('downloads.html', files=result_files)

if __name__ == '__main__':
    app.run(debug=True)
