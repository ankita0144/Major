
from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import base64
from .detectors import bicep_curl

bp = Blueprint('api', __name__)

def decode_image(base64_string):
    img_data = base64.b64decode(base64_string.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@bp.route('/analyze/bicep_curl', methods=['POST'])
def analyze_bicep_curl():
    data = request.get_json()
    image = decode_image(data['image'])
    result = bicep_curl.analyze_bicep_curl(image)
    return jsonify(result)
