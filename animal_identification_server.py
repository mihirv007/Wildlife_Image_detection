from flask import Flask,request,render_template,jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys

#load the model ,model 1.0 accuracy: 90%

model=load_model('animal_model_1.0.keras')

app=Flask(__name__)
CORS(app)

#checking if folder exist 
save_folder='static/save_files_warehouse'
os.makedirs(save_folder,exist_ok=True)

# Health status check route
@app.route("/")
def index():
    return jsonify({
        'status': 'online',
        'message': 'Wildlife Image Detection API is running',
        'model': 'animal_model_1.0.keras',
        'species_supported': ['cat', 'dog', 'elephant', 'horse', 'lion']
    })
@app.route('/api/predict',methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error':'no file part'}),400
    
    file=request.files['file']

    if file.filename=='':
        return jsonify({'error':'file not selected'}),400
    
    if file:
        #save the file
        save_path=os.path.join(save_folder,file.filename)
        file.save(save_path)

        #load and preprocess the file
        image_size=(224,224)
        test_image=image.load_img(save_path,target_size=image_size)
        test_image=image.img_to_array(test_image)

        #normalize the image
        test_image=test_image/255.0
        test_image=np.expand_dims(test_image,axis=0)

        #make prediction
        prediction=model.predict(test_image)
        prediction_class=np.argmax(prediction)
        confidence=np.max(prediction)*100

        class_of_animals={
            0:'cat',
            1:'dog',
            2:'elephant',
            3:'horse',
            4:'lion'
        }

        prediction_label=class_of_animals[prediction_class]

        return jsonify({
            'species':prediction_label,
            'confidence':f'{confidence:.2f}',
            'image_file':file.filename
        })

if __name__=='__main__':
    app.run(debug=True)







