from flask import Flask, render_template, request, url_for
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model('cat_dog_classifier.h5')

app = Flask(__name__)


def predict_image(img_path):
    # Load and preprocess the image
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Normalize the image

    # Make the prediction
    result = model.predict(test_image)

    # Return the prediction
    return 'dog' if result[0][0] >= 0.5 else 'cat'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('index.html', prediction=prediction, img_path=file_path)
    return render_template('index.html', prediction=None, img_path=None)


if __name__ == '__main__':
    app.run(debug=True)
