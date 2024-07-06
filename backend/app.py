from flask import Flask, request, jsonify
from skimage.io import imread
import numpy as np
import joblib
import tempfile
import os
from flask_cors import CORS
from io import BytesIO
import base64
import cv2
import PIL.Image

app = Flask(__name__)
CORS(app)

clf = joblib.load('sourcecameramodel.pkl')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image):
    a0 = image[:, :, 0].mean()
    a1 = image[:, :, 1].mean()
    a2 = image[:, :, 2].mean()
    s0 = image[:, :, 0].std()
    s1 = image[:, :, 1].std()
    s2 = image[:, :, 2].std()
    return np.array([[a0, a1, a2, s0, s1, s2]])

def random_crop_fft(img, W):
    nr, nc = img.shape[:2]
    r1, c1 = np.random.randint(nr-W), np.random.randint(nc-W) 
    imgc = img[r1:r1+W, c1:c1+W, :]

    img1 = imgc - cv2.GaussianBlur(imgc, (3,3), 0)
    imgs1 = np.sum(img1, axis=2)
    
    sf = np.stack([
         np.fft.fftshift(np.fft.fft2( imgs1 )),
         np.fft.fftshift(np.fft.fft2( img1[:,:,0] - img1[:,:,1] )),
         np.fft.fftshift(np.fft.fft2( img1[:,:,1] - img1[:,:,2] )),
         np.fft.fftshift(np.fft.fft2( img1[:,:,2] - img1[:,:,0] )) ], axis=-1)
    return np.abs(sf)

def imread_residual_fft(file, W):
    img = np.array(PIL.Image.open(file)).astype(np.float32) / 255.0
    return random_crop_fft(img, W)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        img_path = 'temp.jpg'
        img_file = file
        img_file.save(img_path)
        image = imread(img_path)
        image_features = extract_features(image)
        predicted_camera = clf.predict(image_features)[0]

        # Calculate noise pattern
        img_np = imread_residual_fft(img_path, W=128)
        noise_pattern = np.mean(img_np, axis=-1)
        noise_pattern_scaled = (255 * (noise_pattern - np.min(noise_pattern)) / (np.max(noise_pattern) - np.min(noise_pattern))).astype(np.uint8)
        noise_pattern_image = PIL.Image.fromarray(noise_pattern_scaled)
        predicted_proba = clf.predict_proba(image_features)[0]
        print(float(predicted_proba.max()))
        # Save noise pattern image and get its path
        noise_pattern_path = 'noise_pattern.jpg'
        noise_pattern_image.save(noise_pattern_path)

        os.remove(img_path)
        return jsonify({'predicted_camera': predicted_camera, 'noise_pattern_path': noise_pattern_path, 'confidence': float(predicted_proba.max())})
    else:
        return jsonify({'error': 'File not allowed'})


def get_dataset_info():
    dataset_path = 'E:\\cameraidentification\\dataset\\train'  # Adjust the path accordingly
    dataset_info = []

    # Loop through each directory (model) in the dataset path
    for model_dir in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, model_dir)):
            model_info = {
                'name': model_dir,
                'count': len(os.listdir(os.path.join(dataset_path, model_dir)))
            }
            dataset_info.append(model_info)

    return dataset_info

# Route to get dataset information
@app.route('/dataset')
def dataset():
    dataset_info = get_dataset_info()
    return jsonify(dataset_info)

if __name__ == '__main__':
    app.run(debug=True)
