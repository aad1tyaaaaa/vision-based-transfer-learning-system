# Vision-Based Transfer Learning System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

A comprehensive implementation of a vision-based transfer learning system using TensorFlow and FastAPI. This project demonstrates the end-to-end process of building, training, optimizing, and deploying a machine learning model for image classification, from data collection to production-ready inference.

## Features

- **Transfer Learning**: Utilizes pre-trained models (MobileNetV2) for efficient training on custom datasets.
- **Data Augmentation**: Implements robust data pipelines with various augmentations for improved model generalization.
- **Model Optimization**: Converts models to TensorFlow Lite for optimized inference on edge devices.
- **RESTful API**: Provides a FastAPI-based web service for real-time image classification.
- **Evaluation Metrics**: Includes confusion matrix, per-class metrics, and performance benchmarks.
- **Modular Design**: Organized codebase with separate scripts for each phase of the ML pipeline.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vision-based-transfer-learning-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
vision-based-transfer-learning-system/
├── data/
│   ├── raw/                 # Raw dataset storage
│   └── processed/           # Processed data
├── models/                  # Trained models and checkpoints
├── scripts/                 # Training and utility scripts
│   ├── data_collection.py   # Dataset download and preparation
│   ├── data_pipeline.py     # Data preprocessing and augmentation
│   ├── train_baseline.py    # Baseline model training
│   ├── fine_tune.py         # Model fine-tuning and evaluation
│   └── optimize_model.py    # Model optimization to TFLite
├── api/
│   └── main.py              # FastAPI application
├── docs/
│   └── benchmark.py         # Benchmarking script
├── notebooks/               # Jupyter notebooks for experimentation
├── .vscode/                 # VS Code configuration
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore rules
```

## Development Phases

The project follows a structured ML development pipeline:

1. **Problem Definition & Dataset Collection**: Define classification classes and acquire labeled image data.
2. **Data Pipeline & Augmentation**: Build efficient data loading pipelines with augmentation techniques.
3. **Baseline Model & Transfer Learning**: Set up pre-trained models and train classification heads.
4. **Fine-tuning & Evaluation**: Unfreeze layers for fine-tuning and perform comprehensive evaluation.
5. **Optimization for Inference**: Convert to TensorFlow Lite with quantization for deployment.
6. **Serving & API**: Develop RESTful API for model inference.
7. **Documentation & Benchmarks**: Generate performance reports and documentation.

## Usage

### Training the Model

Execute the training scripts in sequence:

```bash
python scripts/data_collection.py
python scripts/train_baseline.py
python scripts/fine_tune.py
python scripts/optimize_model.py
```

### Running the API

Start the FastAPI server:

```bash
python api/main.py
```

Or use the VS Code task: `Tasks: Run Task > Run API`

The API will be accessible at `http://localhost:8000`.

### API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Image classification
  - **Input**: Multipart form data with `file` field containing image
  - **Output**: JSON with predicted class, confidence, and probabilities

Example request using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

Example response:

```json
{
  "class": "cat",
  "confidence": 0.95,
  "probabilities": {
    "cat": 0.95,
    "dog": 0.05
  }
}
```

## Benchmarks

Run performance benchmarks:

```bash
python docs/benchmark.py
```

This script measures model size and average inference latency. Update the script with your test image paths for accurate results.

## Deliverables

- **Labeled Dataset**: Cats vs Dogs dataset from TensorFlow Datasets
- **Trained Model**: Keras model saved as `models/fine_tuned_model.h5`
- **Optimized Model**: TensorFlow Lite model at `models/model.tflite`
- **Inference API**: Production-ready FastAPI application
- **Benchmark Report**: Performance metrics generated by benchmark script

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow team for the excellent ML framework
- FastAPI for the modern web framework
- TensorFlow Datasets for easy data access
