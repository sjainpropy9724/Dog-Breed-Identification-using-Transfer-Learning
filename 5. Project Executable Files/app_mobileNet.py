from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import os
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

app = Flask(__name__)

model_path = 'dogbreed_mobilenet.h5'
model = tf.keras.models.load_model(model_path)

UPLOAD_FOLDER = './uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_labels = ['boston_bull', 'dingo', 'pekinese', 'bluetick', 'golden_retriever', 'bedlington_terrier', 'borzoi', 'basenji', 'scottish_deerhound', 'shetland_sheepdog', 'walker_hound', 'maltese_dog', 'norfolk_terrier', 'african_hunting_dog', 'wire-haired_fox_terrier', 'redbone', 'lakeland_terrier', 'boxer', 'doberman', 'otterhound', 'standard_schnauzer', 'irish_water_spaniel', 'black-and-tan_coonhound', 'cairn', 'affenpinscher', 'labrador_retriever', 'ibizan_hound', 'english_setter', 'weimaraner', 'giant_schnauzer', 'groenendael', 'dhole', 'toy_poodle', 'border_terrier', 'tibetan_terrier', 'norwegian_elkhound', 'shih-tzu', 'irish_terrier', 'kuvasz', 'german_shepherd', 'greater_swiss_mountain_dog', 'basset', 'australian_terrier', 'schipperke', 'rhodesian_ridgeback', 'irish_setter', 'appenzeller', 'bloodhound', 'samoyed', 'miniature_schnauzer', 'brittany_spaniel', 'kelpie', 'papillon', 'border_collie', 'entlebucher', 'collie', 'malamute', 'welsh_springer_spaniel', 'chihuahua', 'saluki', 'pug', 'malinois', 'komondor', 'airedale', 'leonberg', 'mexican_hairless', 'bull_mastiff', 'bernese_mountain_dog', 'american_staffordshire_terrier', 'lhasa', 'cardigan', 'italian_greyhound', 'clumber', 'scotch_terrier', 'afghan_hound', 'old_english_sheepdog', 'saint_bernard', 'miniature_pinscher', 'eskimo_dog', 'irish_wolfhound', 'brabancon_griffon', 'toy_terrier', 'chow', 'flat-coated_retriever', 'norwich_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'english_foxhound', 'gordon_setter', 'siberian_husky', 'newfoundland', 'briard', 'chesapeake_bay_retriever', 'dandie_dinmont', 'great_pyrenees', 'beagle', 'vizsla', 'west_highland_white_terrier', 'kerry_blue_terrier', 'whippet', 'sealyham_terrier', 'standard_poodle', 'keeshond', 'japanese_spaniel', 'miniature_poodle', 'pomeranian', 'curly-coated_retriever', 'yorkshire_terrier', 'pembroke', 'great_dane', 'blenheim_spaniel', 'silky_terrier', 'sussex_spaniel', 'german_short-haired_pointer', 'french_bulldog', 'bouvier_des_flandres', 'tibetan_mastiff', 'english_springer', 'cocker_spaniel', 'rottweiler']

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = preprocess_input(x)
    preds = model.predict(np.array([x]))
    predicted_class = class_labels[np.argmax(preds)]

    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            predicted_class= model_predict(file_path, model)
            return redirect(url_for('output', result=predicted_class, file_path=file_path))
    return render_template('predict.html')

@app.route('/output')
def output():
    result = request.args.get('result')
    file_path = request.args.get('file_path')
    return render_template('output.html', result=result, file_path=file_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=False)
