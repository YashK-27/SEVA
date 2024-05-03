import pickle
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


from werkzeug.utils import secure_filename
import os
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torchvision import transforms
import CNN  # Make sure this is the correct import for your CNN model

application = Flask(__name__)

app = application

#Route for home page

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/crop-recommendation', methods = ['GET', 'POST'])
def crop_recommendation():
    if request.method == 'GET':
        return render_template('crop_recommend.html')
    else:
        # reading all the values
        data = CustomData(
            humidity=request.form.get('humidity'),
            temperature=request.form.get('temperature'),
            ph=request.form.get('ph'),
            rainfall=request.form.get('rainfall'),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('crop_recommend.html',results=results[0])
    


disease_info = pd.read_csv('artifacts/disease_info.csv', encoding='cp1252')
model = CNN.CNN(39)  # Make sure this aligns with your saved model
model.load_state_dict(torch.load("artifacts/plant_disease_model_1_latest.pt", map_location=torch.device('cpu')))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = transforms.ToTensor()(image)
    input_data = input_data.unsqueeze(0)
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

@app.route('/disease-detection', methods=['GET', 'POST'])
def disease_detection():
    if request.method == 'GET':
        return render_template('disease_detection.html')
    elif request.method == 'POST':
        if 'image' not in request.files:
            return render_template('disease_detection.html', error='No file part')
        image = request.files['image']
        if image.filename == '':
            return render_template('disease_detection.html', error='No selected file')
        if image:
            filename = secure_filename(image.filename)
            file_path = os.path.join('static/uploads', filename)
            os.makedirs('static/uploads', exist_ok=True)
            image.save(file_path)
            pred_index = prediction(file_path)
            context = {
                'title': disease_info.iloc[pred_index]['disease_name'],
                'description': disease_info.iloc[pred_index]['description'],
                'prevent': disease_info.iloc[pred_index]['Possible Steps'],
                'image_url': disease_info.iloc[pred_index]['image_url']
            }
            return render_template('submit.html', **context)
    else:
        return redirect(url_for('disease_detection'))

@app.route('/crop-info', methods = ['GET'])
def crop_info():
    return render_template('crop_info.html')

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)