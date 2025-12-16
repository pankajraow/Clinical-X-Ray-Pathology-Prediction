from flask import Flask, render_template, request, jsonify, url_for
import os
import time
from xray_service import xray_service

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/heatmaps', exist_ok=True)

@app.before_request
def start_model():
    if xray_service.model is None:
        xray_service.load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Preprocess
            img_tensor, prep_time = xray_service.preprocess(filepath)
            
            # 2. Predict
            predictions, infer_time = xray_service.predict(img_tensor)
            
            # 3. Explain (Top class)
            # Find index of top class
            # model.pathologies is a list of names. predictions is a sorted list of dicts.
            # We need the index corresponding to the top predicted label
            top_label = predictions[0]['label']
            target_idx = xray_service.model.pathologies.index(top_label)
            
            heatmap_path = xray_service.explain(img_tensor, target_idx=target_idx)
            
            # Add Latency Metrics to result page mostly via printing or simple display if we added it to template
            # For now passing them in console
            print(f"Preprocess: {prep_time*1000:.2f}ms, Inference: {infer_time*1000:.2f}ms")
            
            return render_template(
                'result.html', 
                image_url=filepath.replace('\\', '/'), # Fix windows paths for browser
                heatmap_url=heatmap_path.replace('\\', '/'), 
                predictions=predictions
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            img_tensor, prep_time = xray_service.preprocess(filepath)
            predictions, infer_time = xray_service.predict(img_tensor)
            
            return jsonify({
                'predictions': predictions,
                'latency': {
                    'preprocess_ms': round(prep_time * 1000, 2),
                    'inference_ms': round(infer_time * 1000, 2)
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    xray_service.load_model()
    app.run(debug=True, port=5000)
