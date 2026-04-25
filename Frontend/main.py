import os
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

# Load your TensorFlow model
# model = load_model('plantclassificationmodel.h5')
# model = load_model('plantclassificationmodelUpdated.h5')
model = load_model('plantclassificationmodelbest.h5')


# Define a route for the upload page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'image' not in request.files:
            return 'No file part'
        
        file = request.files['image']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return 'No selected file'
        
        if file:
            # Save the file to a location on your server
            
            file_path = 'uploads/' + file.filename
            # file_path = 'Frontend/uploads/' + file.filename
            file.save(file_path)
            
            # Perform prediction
            prediction = predict_image(file_path)
            
            # Delete the uploaded file
            os.remove(file_path)
            
            return prediction
    
    # Render the upload form template if it's a GET request
    return render_template('index.html')

def predict_image(file_path):
    # Load and preprocess the image
    # img = image.load_img(file_path, target_size=(200,300), color_mode='grayscale')
    img = image.load_img(file_path, target_size=(200,300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)[0]
    
    # Post-processing of prediction (e.g., get class label from prediction)
    # For example, if your model outputs probabilities for each class:
    class_label = np.argmax(prediction)
     # Convert to percentage and take the integral part
    class_names=['Basil', 'Chinar', 'Guava', 'Jamun', 'Lemon', 'Mango']

    accuracy = {class_names[i]: round(prediction[i] * 100, 2) for i in range(len(class_names))}

    result=class_names[class_label]

    if accuracy[result] >= 100:
        return "Can't be classified"
    
    # Return the predicted class label and accuracy on separate lines
    return f'<mark><em>{result}</em></mark> <p><strong>Accuracy Probability: </strong>{accuracy}</p>'

if __name__ == '__main__':
    app.run(debug=True)