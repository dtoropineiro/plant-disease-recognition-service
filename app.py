from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from model import ResNet9

app = Flask(__name__)

classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
           'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
           'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
           'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
           'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
           'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
           'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
           'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

PATH = 'plant-disease-state-model.pth'
model = ResNet9(3, 38)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()


# Preprocess the image
def preprocess_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'file is required'}), 400
    file = request.files['file']
    img_bytes = file.read()
    img = preprocess_image(img_bytes)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
        predicted_class = predicted.item()
    return jsonify({'class_id': predicted_class, 'class_name': classes[predicted_class]})


if __name__ == '__main__':
    app.run(debug=True)
