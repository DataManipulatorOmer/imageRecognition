from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

# Flask application
app = Flask(__name__)
model = MobileNetV2(weights='imagenet')

# Define a route for image recognition
@app.route('/image-recognition', methods=['POST'])
def image_recognition():
    # Check if an image file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400

    # Get the uploaded file
    file = request.files['file']

    # Load and preprocess the image
    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
    imageArray = image.img_to_array(img)
    imageArray = np.expand_dims(imageArray, axis=0)
    processedImage = preprocess_input(imageArray)

    predictions = model.predict(processedImage)
    decodedPredictions = decode_predictions(predictions, top=3)[0]

    results = []
    for pred in decodedPredictions:
        label = pred[1]
        confidence = float(pred[2])
        results.append({'label': label, 'confidence': confidence})

    return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=True)
