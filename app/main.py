import logging

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

from AI_Python.model_inferencing import ClassificationModelInference
from AI_Python.recommendation import recommend

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

bag_classification_model = ClassificationModelInference()


@app.route('/')
def index():
    return "<h1>BagTracker Classification</h1>"


@app.route('/api/predict', methods=['POST'])
def predict_bag():
    if request.method == 'POST':
        try:
            f = request.files['file']
            img = Image.open(f.stream)
            img = img.resize((224, 224))
            result = bag_classification_model.model_predict_main(np.array(img.convert('RGB')), 1)
            return jsonify(result)
        except Exception as error:
            logger.exception(error)
            return jsonify({'message': 'Failed to recognize the image'}), 500

    return jsonify({'message': 'Only POST method allowed'}), 400

@app.route('/api/recommend', methods=['GET'])
def recommend_bag():
    if request.method == 'GET':
        try:
            key = request.args.get('model')
            res = recommend(key)
            return jsonify(res)
        except Exception as error:
            logger.exception(error)
            return jsonify({'message': 'Failed to recommend related master bags'}), 500
    return jsonify({'message': 'Only GET method allowed'}), 400
    
if __name__ == "__main__":
    app.run()
