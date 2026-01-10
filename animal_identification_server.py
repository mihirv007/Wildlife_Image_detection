from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

#load the model ,model 1.0 accuracy: 90%

model=load_model('/Users/mihirverma/Animal_indentification/animal_model_1.0.keras')

app=Flask(__name__)

#checking if folder exist 
save_folder='/Users/mihirverma/Animal_indentification/static/save_files_warehouse'
os.makedirs(save_folder,exist_ok=True)

#this route render the html form

@app.route("/")
def upload_file():
    return render_template('animal_indentification.html')

@app.route('/upload',methods=['POST'])

def upload_image():
    if 'file' not in request.files:
        return 'no file part'
    
    file=request.files['file']

    if file.filename=='':
        return 'file not selected'
    
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
        print(prediction)
        
        #result

        prediction_class=np.argmax(prediction)

        confidence=np.max(prediction)*100
        print(confidence)

        class_of_animals={
            0:'cat',
            1:'dog',
            2:'elephant',
            3:'horse',
            4:'lion'
        }

        prediction_label=class_of_animals[prediction_class]

        print('predicted class:',prediction_label)

        return render_template(
            'animal_indentification.html',
            prediction=prediction_label,
            Confidence=f'{confidence:.2f}',
            image_file=file.filename
            )
    
if __name__=='__main__':
    app.run(debug=True)







