import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model("soyabean_model.h5")

class_names = [
    'bacterial_blight',
    'brown_spot',
    'caterpillar',
    'crestamento',
    'diabrotica_speciosa',
    'ferrugen',
    'healthy',
    'mosaic_virus',
    'powdery_mildew',
    'septoria',
    'southern_blight',
    'sudden_death_syndrome',
    'yellow_mosaic'
]

medicine_dict = {
    "bacterial_blight": "Copper Oxychloride 3g per liter.",
    "brown_spot": "Mancozeb 2g per liter.",
    "caterpillar": "Emamectin Benzoate 0.4g per liter.",
    "crestamento": "Carbendazim 1g per liter.",
    "diabrotica_speciosa": "Neem oil spray.",
    "ferrugen": "Propiconazole 1ml per liter.",
    "healthy": "No disease detected.",
    "mosaic_virus": "Remove infected plants.",
    "powdery_mildew": "Sulfur 2g per liter.",
    "septoria": "Chlorothalonil 2g per liter.",
    "southern_blight": "Carbendazim soil treatment.",
    "sudden_death_syndrome": "Improve drainage.",
    "yellow_mosaic": "Control whiteflies."
}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    confidence = round(100 * np.max(prediction), 2)
    predicted_class = class_names[np.argmax(prediction)]
    medicine = medicine_dict[predicted_class]

    return render_template(
        'result.html',
        prediction=predicted_class,
        medicine=medicine,
        confidence=confidence,
        image_path=filepath
    )


if __name__ == '__main__':
    app.run(debug=True)