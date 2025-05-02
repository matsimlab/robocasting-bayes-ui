# Robocasting Experiments Optimization

An interactive web application for predicting, optimizing, and managing robocasting/3D printing experiments using Gaussian Process Regression.

## Overview

This application helps researchers and engineers optimize their robocasting experiments by:

- Predicting print dimensions (width and height) based on processing parameters
- Suggesting optimal experiment parameters to achieve desired dimensions
- Tracking and visualizing experimental data
- Efficiently exploring the design space with Bayesian optimization

## Features

- **Data Management**: Store, visualize, and analyze experimental results
- **Dimension Prediction**: Use Gaussian Process Regression to predict print dimensions with uncertainty quantification
- **Experiment Suggestion**: Get intelligent recommendations for next experiments
- **Design Space Exploration**: Generate diverse experiment points to explore parameter space
- **User Authentication**: Secure multi-user access with admin management
- **Interactive Visualization**: Understand parameter relationships through interactive plots

## Installation

### Docker Installation (Recommended)

The easiest way to run the application is using Docker:

```shell
# Pull and run the container
docker run -d --name robocasting -p 8501:8501 -v robocasting_data:/app/data --restart unless-stopped nazarmedykh/robocasting:v2
```

Then access the application at http://localhost:8501

### Local Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```shell
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```
4. Run the application:
   ```shell
   streamlit run app.py
   ```

## Usage

1. **Login**: Access the application with default credentials (admin/robocasting) on first run
2. **Data Explorer**: View existing data, visualize parameter relationships
3. **Predictions**: Input parameters to predict printed dimensions with confidence intervals
4. **Next Experiment**: Get suggestions for optimal next experiment parameters
5. **Add New Data**: Add new experimental results to the database

## Building the Docker Image

To build and push the Docker image for multiple platforms:

```shell
docker buildx build --platform linux/amd64,linux/arm64 -t nazarmedykh/robocasting:v2 --push .
```

## Architecture

- **Frontend**: Streamlit web application
- **Backend**: Python with scikit-learn and scikit-optimize
- **Database**: SQLite (persistent through Docker volume)
- **ML Models**: Gaussian Process Regression with Bayesian hyperparameter optimization

## Parameter Descriptions

- **Temperature**: Ambient temperature during printing (°C)
- **Humidity**: Ambient humidity (0-1)
- **Layer Height**: Slicer-defined layer height setting
- **Layer Width**: Slicer-defined layer width setting
- **Nozzle Speed**: Print head movement speed
- **Extrusion Multiplier**: Filament extrusion rate multiplier

## Contributing

Contributions to improve the application are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Insert appropriate license information here]
