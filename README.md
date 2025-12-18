# Clinical X-Ray Pathology Prediction

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/pankajraow/Clinical-X-Ray-Pathology-Prediction)

A Flask-based web application that serves pretrained models from [TorchXRayVision](https://github.com/mlmed/torchxrayvision) to predict pathologies from Chest X-Rays.

## Features
- **Pathology Prediction**: Uses `densenet121-res224-all` to predict probabilities for multiple pathologies.
- **Explainability**: Generates Grad-CAM/Saliency heatmaps to visualize the area of interest.
- **DICOM Support**: Automatically handles and de-identifies DICOM images.
- **REST API**: Provides JSON endpoints for programmatic access.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pankajraow/Clinical-X-Ray-Pathology-Prediction.git
   cd Clinical-X-Ray-Pathology-Prediction
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask Application:
   ```bash
   python app.py
   ```
   The app will start on `http://127.0.0.1:5000`.

2. **Web Interface**: Open your browser to `http://127.0.0.1:5000`.
   - Upload a JPG, PNG, or DICOM image.
   - View predictions and heatmaps.

3. **API Usage**:
   ```bash
   curl -X POST -F "file=@sample_xray.jpg" http://127.0.0.1:5000/api/predict
   ```

## Project Structure
- `app.py`: Main Flask application.
- `xray_service.py`: Model loading, preprocessing, and inference logic.
- `templates/`: HTML templates.
- `static/`: CSS and uploaded images.
