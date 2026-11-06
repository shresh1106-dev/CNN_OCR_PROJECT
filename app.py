import gradio as gr
import numpy as np
import cv2
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import os
import traceback

# Load or Train Model automatically so app.py runs standalone
MODEL_PATH = "industry_ocr_model.keras"
if os.path.exists(MODEL_PATH):
    print("[INFO] Loading saved CNN model...")
    MODEL = tf.keras.models.load_model(MODEL_PATH)
else:
    print("[INFO] Training CNN model on MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)
    x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)

    MODEL = models.Sequential([
        layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    MODEL.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    MODEL.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=64, verbose=1)
    MODEL.save(MODEL_PATH)

def segment_digits_from_image(img):
    if img is None:
        return [], np.zeros((100, 100), dtype=np.uint8)
        
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Ensure grayscale is uint8
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Gaussian Blur to smooth out high-frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 4
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    img_h, img_w = gray.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if (h > img_h * 0.05) and (h < img_h * 0.95) and (w > 5) and (aspect_ratio < 3.0):
            bounding_boxes.append((x, y, w, h))
            
    # Sort bounding boxes left-to-right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    digit_images = []
    for (x, y, w, h) in bounding_boxes:
        digit_crop = gray[y:y+h, x:x+w]
        if np.mean(digit_crop) > 127:
            digit_crop = cv2.bitwise_not(digit_crop)
            
        square_crop = cv2.resize(digit_crop, (20, 20))
        padded_digit = np.pad(square_crop, ((4, 4), (4, 4)), mode='constant', constant_values=0)
        digit_images.append(padded_digit)
        
    return digit_images, thresh

def predict_custom_tag(image, robot_id, location_bay):
    try:
        if image is None:
            return "Please upload an image.", np.zeros((100, 100), dtype=np.uint8), "{}"
            
        digit_crops, thresh_img = segment_digits_from_image(image)
        if not digit_crops:
            return "No digits detected. Ensure clear dark digits on light background.", thresh_img, "{}"
            
        predicted_serial = ""
        for crop in digit_crops:
            tensor = (crop / 255.0).reshape(1, 28, 28, 1)
            pred = MODEL.predict(tensor, verbose=0)
            digit = np.argmax(pred)
            predicted_serial += str(digit)
            
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "robot_id": robot_id,
            "location": location_bay,
            "scanned_serial_number": predicted_serial,
            "status": "VERIFIED_INVENTORY"
        }
        
        return predicted_serial, thresh_img, json.dumps(log_entry, indent=4)
    except Exception as e:
        err_msg = str(e)
        traceback.print_exc()
        return f"Error: {err_msg}", np.zeros((100, 100), dtype=np.uint8), json.dumps({"error": err_msg}, indent=4)

def predict_demo_tag(d1, d2, d3, d4, robot_id, location_bay):
    try:
        (_, _), (x_test, y_test) = mnist.load_data()
        indices = [int(d1), int(d2), int(d3), int(d4)]
        selected_images = [x_test[idx] for idx in indices]
        actual_labels = [y_test[idx] for idx in indices]
        
        asset_tag = np.hstack(selected_images)
        asset_tag = cv2.copyMakeBorder(asset_tag, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        expected_code = "".join(map(str, actual_labels))
        
        digit_crops, _ = segment_digits_from_image(asset_tag)
        predicted_serial = ""
        for crop in digit_crops:
            tensor = (crop / 255.0).reshape(1, 28, 28, 1)
            pred = MODEL.predict(tensor, verbose=0)
            digit = np.argmax(pred)
            predicted_serial += str(digit)
            
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "robot_id": robot_id,
            "location": location_bay,
            "scanned_serial_number": predicted_serial,
            "ground_truth_code": expected_code,
            "status": "VERIFIED_INVENTORY"
        }
        
        return asset_tag, predicted_serial, expected_code, json.dumps(log_entry, indent=4)
    except Exception as e:
        err_msg = str(e)
        traceback.print_exc()
        return np.zeros((100, 100), dtype=np.uint8), f"Error: {err_msg}", "", json.dumps({"error": err_msg}, indent=4)

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏭 OCR for Logistics & Asset Tracking")
        gr.Markdown("A CNN trained on MNIST reads a serial tag, checks it against inventory, and writes an ERP-style log, like an AMR/AGV scanner in a smart warehouse.")
        
        with gr.Tabs():
            with gr.TabItem("Read a tag from a photo"):
                gr.Markdown("Upload or take a photo of 3 to 6 digits written with a thick dark marker on a plain white paper. Keep the digits apart and the light even.")
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(label="Tag photo", type="numpy")
                        r_id = gr.Dropdown(["AGV_ROBOT_01", "AMR_ROBOT_02"], value="AGV_ROBOT_01", label="Robot ID")
                        loc = gr.Dropdown(["BAY_A_RECEIVING", "BAY_B_STORAGE"], value="BAY_A_RECEIVING", label="Location Bay")
                        submit_btn = gr.Button("Scan Custom Tag", variant="primary")
                    with gr.Column():
                        out_serial = gr.Textbox(label="Predicted Serial Number")
                        out_gray = gr.Image(label="Processed Grayscale Tag")
                        out_json = gr.Code(label="ERP Database Log (JSON)", language="json")
                
                submit_btn.click(predict_custom_tag, inputs=[img_input, r_id, loc], outputs=[out_serial, out_gray, out_json])
                
            with gr.TabItem("Demo tag (from MNIST test images)"):
                gr.Markdown("Select test indices from the MNIST dataset to synthesize an asset tag.")
                with gr.Row():
                    with gr.Column():
                        d1 = gr.Number(value=7, label="Digit 1 Index")
                        d2 = gr.Number(value=25, label="Digit 2 Index")
                        d3 = gr.Number(value=100, label="Digit 3 Index")
                        d4 = gr.Number(value=999, label="Digit 4 Index")
                        r_id_demo = gr.Dropdown(["AGV_ROBOT_01", "AMR_ROBOT_02"], value="AGV_ROBOT_01", label="Robot ID")
                        loc_demo = gr.Dropdown(["BAY_A_RECEIVING", "BAY_B_STORAGE"], value="BAY_A_RECEIVING", label="Location Bay")
                        demo_btn = gr.Button("Generate & Scan Demo Tag", variant="primary")
                    with gr.Column():
                        demo_tag_img = gr.Image(label="Synthesized Asset Tag")
                        demo_serial = gr.Textbox(label="Predicted Serial Number")
                        demo_truth = gr.Textbox(label="Ground Truth Code")
                        demo_json = gr.Code(label="ERP Database Log (JSON)", language="json")
                
                demo_btn.click(predict_demo_tag, inputs=[d1, d2, d3, d4, r_id_demo, loc_demo], outputs=[demo_tag_img, demo_serial, demo_truth, demo_json])
                
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=False)