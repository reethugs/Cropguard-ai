import gradio as gr
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image

model = tf.saved_model.load('savedmodel')
infer = model.signatures['serving_default']

class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
               'Tomato_Bacterial_spot', 'Tomato_Early_blight',
               'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
               'Tomato__Tomato_YellowLeaf__Curl_Virus']

treatments = {
    'Pepper__bell___Bacterial_spot': 'Apply copper based fungicide. Remove infected leaves immediately.',
    'Pepper__bell___healthy': 'Plant is healthy! Keep watering regularly.',
    'Tomato_Bacterial_spot': 'Use copper spray. Avoid overhead watering.',
    'Tomato_Early_blight': 'Apply fungicide every 7-10 days. Remove infected leaves.',
    'Tomato_healthy': 'Plant is healthy! No action needed.',
    'Tomato_Late_blight': 'Apply chlorothalonil fungicide immediately.',
    'Tomato_Leaf_Mold': 'Improve air circulation. Apply fungicide.',
    'Tomato_Septoria_leaf_spot': 'Remove infected leaves. Apply mancozeb fungicide.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Use miticide spray. Keep plants well watered.',
    'Tomato__Target_Spot': 'Apply copper fungicide. Remove infected leaves.',
    'Tomato__Tomato_mosaic_virus': 'No cure. Remove infected plants to prevent spread.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'No cure. Control whiteflies to prevent spread.'
}


custom_css = """
body {
    background-image: url('https://tse2.mm.bing.net/th/id/OIP.WxTJG3_OV7cS49I9UlcDVgHaEo?pid=Api&P=0&h=180') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
}
.gradio-container {
    background: transparent !important;
    border-radius: 0px !important;
    padding: 20px !important;
    max-width: 800px !important;
    margin: 20px auto !important;
}
.gr-panel, .gr-box, .gr-form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
label, .gr-block {
    color: white !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.7) !important;
}
.gr-button-primary {
    background-color: #2e7d32 !important;
    border: none !important;
    color: white !important;
}
.gr-button-primary:hover {
    background-color: #1b5e20 !important;
}
footer {
    display: none !important;
}
"""


def predict_disease(image):
    img = Image.fromarray(image).resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    input_tensor = tf.constant(img_array)
    predictions = infer(input_tensor)
    output_key = list(predictions.keys())[0]
    pred_array = predictions[output_key].numpy()[0]

    predicted_class = class_names[np.argmax(pred_array)]
    confidence = round(100 * np.max(pred_array), 2)

    if confidence < 70:
        return "Image not recognized! Please upload a Pepper or Tomato leaf image only.", ""

    treatment = treatments[predicted_class]
    result = f"Disease: {predicted_class.replace('_', ' ')}\nConfidence: {confidence}%"
    return result, treatment

demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(label="Upload Leaf Image"),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Textbox(label="Suggested Treatment")
    ],
    title="🌿 CropGuard AI",
    description="AI-powered crop disease detection for Pepper and Tomato plants. Upload a leaf image to detect disease.",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
    ),
    css=custom_css
)

demo.launch()
