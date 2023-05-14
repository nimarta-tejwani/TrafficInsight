import os
from flask import Flask, render_template, request, redirect, Response
import requests
# from predict import predict, set_path, video_src

app = Flask(__name__)
from flask import send_file
import logging
import subprocess

# Set the logging level to DEBUG
app.logger.setLevel(logging.DEBUG)

# Configure a stream handler to output log messages to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
app.logger.addHandler(stream_handler)


def save_video_from_file(video_file):
    # Create the output filename
    output_file = "output.mp4"

    # Save the uploaded video file to the current directory
    video_file.save(output_file)

    return output_file

@app.route("/upload", methods=["GET"])
def upload_video():
    # Check if a file was uploaded
    if "video" not in request.files:
        return "No video file provided."

    video_file = request.files["video"]

    # Check if the file has a valid filename
    if video_file.filename == "":
        return "Invalid video filename."

    # Save the video file
    saved_file = save_video_from_file(video_file)

    return f"Video saved as '{saved_file}' in the current directory."


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/select_video', methods=['POST'])
def select_video():
    video_file = request.files['video']
    video_path = 'my_video.mp4'  # Provide the path where you want to save the video # Build the absolute file path
    try:
        video_file.save(video_path)
        print(video_file)
        return "Video Saved"
    except Exception as e:
        return "Can't upload"



@app.route('/download', methods=['GET'])
def download_result():
    # File name
    file_name = "my_video.mp4"

    # Send the file as a response
    try:
        return send_file(file_name, as_attachment=True)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
