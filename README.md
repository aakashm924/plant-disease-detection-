# Plant AI

Plant AI is a full-stack plant disease detection and plant-care assistant. Users can upload or capture a leaf image,
get an AI disease prediction, view confidence and care advice, and ask plant-related questions through a chatbot.

## Features

- Detects plant diseases from leaf images using a TensorFlow/Keras CNN model.
- Shows prediction confidence and top possible disease matches.
- Provides disease descriptions, treatment suggestions, and precautions.
- Includes an AI chatbot for plant-care and disease-related questions.
- Supports FAQ-based fast answers with AI fallback through Anthropic Claude.
- React frontend with image upload, camera capture, chatbot widget, and result dashboard.

## Tech Stack

- Frontend: React, TypeScript, Vite, Tailwind CSS
- AI API: Python, Flask, TensorFlow/Keras, Pillow, NumPy
- Chatbot API: Python, Flask, pandas, scikit-learn, Anthropic SDK
- Model: `plant_disease_model.keras`
- Dataset: PlantVillage-style plant disease image dataset

## Project Structure

```text
plant_ai/
+-- CHATBOT/
|   +-- AI/
|   |   +-- app.py
|   |   +-- train_cnn.py
|   |   +-- requirements.txt
|   |   +-- plant_disease_model.keras
|   |   +-- class_indices.json
|   |   +-- dataset/
|   +-- chat/
|       +-- main.py
|       +-- plant_faq.csv
+-- frontend/
|   +-- src/
|   +-- public/
|   +-- package.json
|   +-- vite.config.ts
+-- validate_setup.py
+-- QUICKSTART.md
+-- README.md
```

## Supported Disease Classes

The model supports 38 classes across crops such as apple, cherry, corn, grape, orange, peach, pepper, potato,
squash, strawberry, and tomato. The class list is stored in:

## Prerequisites

Install these before running the project:

- Python 3.10 or newer
- Node.js 18 or newer
- npm
- Git

For chatbot AI responses, set an Anthropic API key:

On Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

## Installation

### 1. Clone the Repository

git clone https://github.com/your-username/plant_ai.git
cd plant_ai


### 2. Install Python Dependencies

cd CHATBOT/AI
pip install -r requirements.txt

The chatbot server uses the same Python environment. If a package is missing, install it with:

```bash
pip install flask flask-cors pandas scikit-learn anthropic
```

### 3. Install Frontend Dependencies

```bash
cd ../../frontend
npm install
```

## Running the Application

Open three terminals from the project root.

### Terminal 1: Start Disease Detection API

```bash
cd CHATBOT/AI
python app.py
```

Runs on:

```text
http://localhost:4000
```

### Terminal 2: Start Chatbot API

```bash
cd CHATBOT/chat
python main.py
```

Runs on:

```text
http://localhost:3000
```

### Terminal 3: Start Frontend

```bash
cd frontend
npm run dev
```

Open the Vite URL in your browser:

```text
http://localhost:5173
```

## API Endpoints

### Disease Detection API

Base URL:

```text
http://localhost:4000
```

Endpoints:

- `POST /predict` - Upload a leaf image and receive disease prediction results.
- `GET /stats` - View scan statistics for the current server session.
- `GET /health` - Check if the AI server is running.

Example `/predict` form field:

```text
file: image file
```

### Chatbot API

Base URL:

```text
http://localhost:3000
```

Endpoints:

- `POST /ask` - Ask a plant-care or disease-related question.
- `POST /suggest` - Get follow-up question suggestions.
- `GET /health` - Check if the chatbot server is running.

Example `/ask` body:

```json
{
  "query": "How do I treat tomato early blight?",
  "session_id": "default"
}
```

## Training the Model

The training script is available at:

```text
CHATBOT/AI/train_cnn.py
```

Make sure the dataset exists in the expected folder structure before training:

```text
CHATBOT/AI/dataset/PlantVillage/train/<class_name>/
```

Then run:

```bash
cd CHATBOT/AI
python train_cnn.py
```

The trained model is saved as:

```text
CHATBOT/AI/plant_disease_model.keras
```

## GitHub Upload Notes

Before uploading to GitHub, avoid committing generated or very large files unless you really need them in the repository.

Recommended files/folders to exclude:

```text
.venv/
node_modules/
frontend/dist/
logs/
__pycache__/
*.pyc
CHATBOT/AI/heatmaps/
```

The dataset and model can be very large:

```text
CHATBOT/AI/dataset/
CHATBOT/AI/plant_disease_model.keras
```

If these files are large, use Git LFS or upload them separately and include download instructions.

Basic GitHub commands:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/plant_ai.git
git push -u origin main
```

## Troubleshooting

### Frontend Cannot Connect to Backend

Make sure both backend servers are running:

- Chatbot API on port `3000`
- Disease Detection API on port `4000`
- Frontend on port `5173`

### Chatbot Returns an API Error

Check that `ANTHROPIC_API_KEY` is set correctly in your terminal before starting `CHATBOT/chat/main.py`.

### Prediction Fails

Check that these files exist:

```text
CHATBOT/AI/plant_disease_model.keras
CHATBOT/AI/class_indices.json
```

Also upload a clear, well-lit leaf image in JPG, PNG, or WEBP format.

### Camera Does Not Open

Camera access requires a secure browser context. Use:

```text
http://localhost:5173
```

or deploy the app with HTTPS.

## Future Improvements

- Add user login and scan history.
- Store predictions in a database.
- Add deployment configuration for cloud hosting.
- Improve Grad-CAM heatmap display in the frontend.
- Add automated tests for backend APIs and frontend components.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
