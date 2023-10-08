from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from vehicle_monitoring import predict
from flask import send_file

app = Flask(__name__)

# Global variable to track processing status
processing = False

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    video = request.files['video']
    if video.filename == '':
        return redirect(url_for('index'))

    video.save("my_video.mp4")

    upload_message = "Upload Successful!"
    return render_template('index.html', upload_message=upload_message)


@app.route('/process')
def process_video():
    predict()
    message = "Video Processed Successfully!"
    table_html, csv_exists = display_output()
    return render_template('index.html', message=message, table_html=table_html, csv_exists=csv_exists)


@app.route('/download', methods=['GET'])
def download_result():
    # File name
    file_name = "output.csv"

    # Send the file as a response
    try:
        return send_file(file_name, as_attachment=True)
    except Exception as e:
        return str(e)


def display_output():
    output_file_path = os.path.join(os.getcwd(), 'output.csv')  # Path to the CSV file
    if os.path.exists(output_file_path):
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(output_file_path)
        # Convert the DataFrame to an HTML table
        table_html = df.to_html(classes='table table-bordered table-striped')
        return table_html, True  # Pass True to indicate that the CSV file exists
    else:
        return "", False 


if __name__ == '__main__':
    app.run(debug=True)
