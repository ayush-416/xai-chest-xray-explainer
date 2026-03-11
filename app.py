from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from natural_explanation import generate_explanation

app = Flask(__name__)

IMG_SIZE = 128

# Load model once
model = tf.keras.models.load_model("heart_xray_cnn.h5")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = img.reshape(1,IMG_SIZE,IMG_SIZE,1)

    pred = model.predict(img)

    class_id = np.argmax(pred)
    confidence = float(pred[0][class_id])

    label = "Cardiomegaly" if class_id==1 else "Normal"

    # Example JSON
    json_output = {
        "prediction": label,
        "confidence": confidence,
        "heart_overlap_average": 0.55,
        "agreement_score": 1.0,
        "trust_level": "Moderate"
    }

    explanation = generate_explanation(json_output)

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "explanation": explanation
    })


if __name__ == "__main__":
    app.run(debug=True)