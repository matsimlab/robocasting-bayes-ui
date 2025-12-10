# Bayesian Optimization Tool for Robocasting Parameter Selection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17886407.svg)](https://doi.org/10.5281/zenodo.17886407)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive web application for optimizing robocasting (direct ink writing) experiments using Gaussian Process Regression and Bayesian optimization. This tool enables systematic parameter selection to minimize dimensional mismatch between target and printed structures.

Authors: Nazarii Mediukh, Vladyslav Naumenko, Anton Krasikov, Vladyslav Bilyi, Oleksandr Vasiliev, Ostap Zgalat-Lozynskyi. 
Institution: Institute for Problems of Materials Science, National Academy of Sciences of Ukraine
---

## Overview

This tool addresses a critical challenge in robocasting: achieving precise dimensional control when slicer settings don't directly translate to printed dimensions due to material behavior, environmental factors, and process dynamics.

**Key capabilities:**
- Train Gaussian Process Regression models on experimental width/height measurements
- Predict dimensional outcomes with uncertainty quantification (95% confidence intervals)
- Suggest optimal next experiment parameters using Bayesian optimization
- Minimize mismatch between target dimensions and actual printed structures
- Track optimization history and link suggestions to experimental results

**Scientific Innovation:** Focuses optimization on the **3 most impactful parameters** identified through machine learning analysis ([Mediukh et al., 2024](https://doi.org/10.5281/zenodo.17782507)): nozzle speed, extrusion multiplier, and layer count. This targeted approach achieves efficient optimization while maintaining prediction quality.

---

## Table of Contents

- [Features](#features)
- [Why Only 3 Parameters?](#why-only-3-parameters)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Docker (Recommended)](#docker-recommended)
  - [Local Development](#local-development)
- [Usage](#usage)
  - [1. Data Management](#1-data-management)
  - [2. Predictions](#2-predictions)
  - [3. Next Experiment](#3-next-experiment)
  - [4. Add New Data](#4-add-new-data)
  - [5. Optimization History](#5-optimization-history)
- [Methodology](#methodology)
  - [Gaussian Process Regression](#gaussian-process-regression)
  - [Bayesian Optimization](#bayesian-optimization)
  - [Model Performance](#model-performance)
- [Parameter Descriptions](#parameter-descriptions)
- [Initial Dataset](#initial-dataset)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Features

- **📊 Data Management**: SQLite database with soft-delete archiving to preserve experimental history
- **🤖 GPR Modeling**: Fixed kernel (ConstantKernel × RBF + WhiteKernel) matching published analysis
- **🎯 Bayesian Optimization**: Two complementary modes:
  - *Single Point*: Pure dimensional accuracy optimization (exploitation)
  - *Design Space Exploration*: Balanced accuracy + diversity (exploration + exploitation)
- **🔒 User Authentication**: Multi-user support with admin controls for collaborative research
- **📈 Interactive Visualization**: Real-time Plotly charts for parameter relationships
- **🐳 Docker Support**: Containerized deployment with persistent data volumes
- **🔗 Suggestion Tracking**: Link experimental results to optimization suggestions for closed-loop validation
- **📉 Uncertainty Quantification**: 95% confidence intervals on all predictions

---

## Repository Structure

```
gpr_wrapper/
├── app.py                      # Main Streamlit application (web interface)
├── models.py                   # GPR training, cross-validation, and prediction
├── database.py                 # SQLite operations (experiments + suggestions)
├── next_point.py               # Bayesian optimization algorithms
├── visualization.py            # Plotly scatter plots and summary statistics
├── auth.py                     # User authentication logic
├── auth_db.py                  # User database management
├── utils.py                    # Helper functions (validation, CSS)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
├── LICENSE                     # MIT License
├── CITATION.cff                # Citation metadata
├── README.md                   # This file
├── cleaned_df.csv              # Initial dataset (auto-imported)
├── data/
│   └── robocasting.db         # SQLite database (generated at runtime)
│   └── robocasting_auth.db    # SQLite database for auth (generated at runtime)
├── static/                     # CSS and styling assets
└── templates/                  # HTML templates (if used)
```

---

## Installation

### Prerequisites
- **Python 3.8+**
- **pip**
- **Docker** (optional, for containerized deployment)

### Docker (Recommended)

Docker provides the easiest deployment with data persistence:

```bash
# Pull and run the container
docker run -d \
  --name robocasting-bayes \
  -p 8501:8501 \
  -v robocasting_data:/app/data \
  --restart unless-stopped \
  nazarmedykh/robocasting:v1.0.0
```

**Access the application:** Open your browser to `http://localhost:8501`

**Default credentials:** Username: `admin`, Password: `robocasting`

**Data persistence:** The `-v robocasting_data:/app/data` flag creates a named Docker volume. Your experimental data survives container restarts and updates.

To stop the container:
```bash
docker stop robocasting-bayes
```

To view logs:
```bash
docker logs robocasting-bayes
```

### Local Development

For development or environments without Docker:

```bash
# Clone the repository
git clone https://github.com/matsimlab/robocasting-bayes-ui.git
cd robocasting-bayes-ui

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

**Access the application:** Streamlit will automatically open `http://localhost:8501` in your browser.

**First run:** On initial startup, the application will:
1. Create `data/robocasting.db` SQLite database
2. Import data from `cleaned_df.csv` (if present)
3. Create default admin user (credentials shown in terminal)

---

## Usage

### 1. Data Management

**Navigate to:** Data Explorer tab

**Features:**
- **View dataset**: Display all experimental records in a table
- **Summary statistics**: Total records, average dimensions, temperature ranges
- **Visualizations**: Scatter plots showing parameter-dimension relationships
  - Width vs Temperature
  - Height vs Temperature
  - Width vs Extrusion Multiplier
  - Height vs Extrusion Multiplier
- **Archive records**: Soft-delete experiments to exclude from model training while preserving data
- **Show archived**: Toggle to view archived records
- **Clear cache**: Force reload if experiencing data issues

**Why archiving matters:** Use archiving to:
- Exclude outliers from model training without deleting data
- Test model performance with/without specific experiments
- Preserve failed experiments for future analysis

### 2. Predictions

**Navigate to:** Predictions tab

**Workflow:**
1. **Model training**: Automatically trains GPR models on all non-archived data
2. **Performance metrics**: View 5-fold cross-validation results (MAE and R² for width/height)
3. **Enter parameters**: Input process parameters using sliders/number inputs
   - Temperature, humidity (environmental)
   - Layer count, layer height, layer width (slicer settings)
   - Nozzle speed, extrusion multiplier (key parameters)
4. **Calculate prediction**: Click button to get:
   - Predicted average width (mm) with ±uncertainty
   - Predicted average height (mm) with ±uncertainty
   - 95% confidence intervals
   - Model kernel and hyperparameters
5. **Transfer to experiments**: Click "Transfer to Add New Data" to use these parameters in a new experiment

**Interpreting results:**
- **MAE (Mean Absolute Error)**: Average prediction error in mm. Lower is better. Typical values: 0.1-0.3 mm.
- **R² Score**: Proportion of variance explained (0-1). Higher is better. Typical values: 0.7-0.95.
- **Uncertainty**: Width of 95% confidence interval. Higher uncertainty suggests the parameter combination is far from training data.

### 3. Next Experiment

**Navigate to:** Next Experiment tab

**Two optimization modes:**

#### Single Point Optimization (Pure Accuracy Focus)
Use when you want **one** next experiment that best matches target dimensions.

**Steps:**
1. **Specify target dimensions**:
   - Target Height (mm per layer): e.g., 1.2 mm if you want 80% of 1.5 mm slicer setting
   - Target Width (mm): e.g., 2.0 mm to match slicer setting
2. **Click "Find Best Dimension Match"**
3. **Review suggestion**:
   - Process parameters (temp, humidity, layer count, etc.)
   - Predicted width/height
   - Dimension mismatch (how far predicted is from target)
   - Height/Width match percentages
4. **Click "Use These Parameters"** to transfer to Add New Data

**Objective:** Minimize `|predicted_height - target_height| + |predicted_width - target_width|`

#### Design Space Exploration (Balanced Accuracy + Diversity)
Use when you want **multiple** diverse experiments that explore the parameter space.

**Steps:**
1. **Specify target dimensions** (same as above)
2. **Select number of suggestions**: 2-10 points (slider)
3. **Click "Generate Suggestions"**
4. **Review suggestions** in tabs (Suggestion 1, 2, 3, ...)
5. **Click "Use Parameters N"** for any suggestion to transfer to Add New Data

**Objective:** Balance dimensional accuracy with exploration (diversity penalty prevents suggesting similar points)

**When to use which mode:**
- **Single Point**: You need one more experiment and want best accuracy
- **Design Space**: You want to explore multiple regions, build intuition, or have capacity for parallel experiments

### 4. Add New Data

**Navigate to:** Add New Data tab

**Workflow:**
1. **Source selection** (optional): Choose a suggestion from dropdown to prefill parameters
2. **Enter measurements** (required): 3 replicate measurements each:
   - Height 1, Height 2, Height 3 (mm)
   - Width 1, Width 2, Width 3 (mm)
3. **Enter parameters** (auto-filled if from suggestion):
   - Temperature (°C)
   - Humidity (%)
   - Layer Count
   - Slicer Layer Height (mm)
   - Slicer Layer Width (mm)
   - Slicer Nozzle Speed
   - Extrusion Multiplier
4. **Submit**: Click "Add Data Point"

**Validation:**
- All measurements must be > 0
- Humidity must be 0-100%
- Temperature must be > 0°C
- Extrusion multiplier must be > 0

**Suggestion tracking:** If you selected a suggestion, the experiment is automatically linked (stored as `suggestion_id`). This enables:
- Tracking which suggestions were executed
- Evaluating optimization performance
- Closed-loop validation of predictions vs. actual outcomes

### 5. Optimization History

**Navigate to:** Optimization History tab

**Features:**
- **View all suggestions**: Every Bayesian optimization suggestion ever generated
- **Filter by**:
  - Dataset size (how many experiments existed when suggestion was made)
  - Suggestion type (single_point vs. design_space_exploration)
- **For each suggestion, see**:
  - Suggested parameters
  - Predicted width/height
  - Dimension mismatch
  - Target dimensions (if specified)
  - Timestamp
- **Quick actions**: Click "Use it" button to transfer parameters to Add New Data
- **Export**: Download history as CSV for external analysis

**Why this matters:** Track how your optimization strategy evolves as you collect more data. Early suggestions might be exploratory (high uncertainty), while later suggestions focus on fine-tuning (low mismatch).

---

## Methodology

### Gaussian Process Regression

**Model Architecture:**
```python
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
model = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-10,              # Numerical stability
    n_restarts_optimizer=5,   # Multiple kernel hyperparameter optimizations
    normalize_y=True,         # Standardize targets
    random_state=42
)
```

**Why this kernel?**
- **ConstantKernel**: Scales overall variance
- **RBF (Radial Basis Function)**: Captures smooth, continuous relationships (nearby parameters → similar outcomes)
- **WhiteKernel**: Models measurement noise and process stochasticity
- This configuration matched best performance in the [robocasting study](https://doi.org/10.5281/zenodo.17782507)

**Training:**
- Two independent models: one for width, one for height
- Features: `[slicer_nozzle_speed, slicer_extrusion_multiplier, layer_count]`
- Targets: `avg_width = mean(width_1, width_2, width_3)`, `avg_height = mean(height_1, height_2, height_3)`
- Feature scaling: StandardScaler (zero mean, unit variance)
- No hyperparameter tuning: Fixed kernel hyperparameters for consistency

### Bayesian Optimization

**Objective Function:**
```
minimize: |predicted_height - target_height| + |predicted_width - target_width|
```

Optional additions:
- **Uncertainty penalty** (design space mode): `+ 0.2 * (width_std + height_std)` encourages exploration
- **Diversity penalty** (design space mode): `+ diversity_weight * exp(-distance/sigma)` prevents redundant suggestions

**Optimization Algorithm:**
- **Library**: `scikit-optimize` (`gp_minimize`)
- **Search space**: 3D continuous/discrete
  - slicer_nozzle_speed: [3.0, 22.0] continuous
  - slicer_extrusion_multiplier: [0.4, 0.8] continuous
  - layer_count: [1, 3] integer
- **Iterations**: 100 evaluations (single point) or 60 per point (design space)
- **Random restarts**: 3 independent optimizations, return best
- **Acquisition function**: Expected Improvement (default in `gp_minimize`)

**Why multiple restarts?** Bayesian optimization can get stuck in local minima. Running 3 independent optimizations with different random seeds increases chances of finding the global optimum.

**Fixed parameters during optimization:**
- `temp`: Set to mean of existing data (or user-specified)
- `humidity`: Set to mean of existing data (or user-specified)
- `slicer_layer_height`: Set to mean of existing data
- `slicer_layer_width`: Set to mean of existing data

Rationale: These parameters have lower predictive importance and are often constrained by lab conditions or equipment.

### Model Performance

**Evaluation:** 5-fold cross-validation on the training dataset

**Metrics:**
- **MAE (Mean Absolute Error)**: Average absolute prediction error (mm)
  - `MAE = mean(|y_true - y_pred|)`
  - Direct measure of typical error magnitude
  - Reported as mean ± std across 5 folds
- **R² Score**: Proportion of variance explained
  - `R² = 1 - SS_res / SS_tot`
  - Range: 0 (no predictive power) to 1 (perfect predictions)
  - Reported as mean ± std across 5 folds

**Interpretation:**
- **Good performance**: MAE < 0.2 mm, R² > 0.8
- **Acceptable**: MAE = 0.2-0.4 mm, R² = 0.6-0.8
- **Poor** (need more data): MAE > 0.4 mm, R² < 0.6

**Uncertainty Quantification:**
- GPR provides posterior standard deviation at each prediction point
- We report 95% confidence interval: `±1.96 * std`
- High uncertainty → prediction point is far from training data (extrapolation)

---

## Parameter Descriptions

| Parameter | Symbol | Range | Unit | Role | Optimized? |
|-----------|--------|-------|------|------|------------|
| **Temperature** | `temp` | 15-30 | °C | Affects material viscosity and drying rate | ❌ Fixed |
| **Humidity** | `humidity` | 30-70 | % | Influences drying speed and interlayer adhesion | ❌ Fixed |
| **Layer Count** | `layer_count` | 1-3 | - | Number of layers (directly affects total height) | ✅ **Optimized** |
| **Slicer Layer Height** | `slicer_layer_height` | 0.5-2.0 | mm | Target height per layer in slicer | ❌ Fixed |
| **Slicer Layer Width** | `slicer_layer_width` | 1.0-3.0 | mm | Target line width in slicer | ❌ Fixed |
| **Nozzle Speed** | `slicer_nozzle_speed` | 3-22 | mm/s | Print head movement speed | ✅ **Optimized** |
| **Extrusion Multiplier** | `slicer_extrusion_multiplier` | 0.4-0.8 | - | Material flow rate multiplier | ✅ **Optimized** |

**Measurement Outputs:**
- **Height** (mm): Total height of printed structure (measured 3 times at different locations)
- **Width** (mm): Line width of printed structure (measured 3 times at different locations)

**Why 3 measurements?** Robocasting prints are rarely perfectly uniform. Taking 3 measurements at different locations and averaging captures within-sample variability and provides more robust targets for GPR training.

---

## Initial Dataset

The repository includes `cleaned_df.csv` containing the initial experimental dataset used to train the models. This dataset is automatically imported to the SQLite database (`data/robocasting.db`) on first run.

**Dataset contents:**
- 58 samples (as per robocasting study)
- 3 replicate measurements per sample (height_1/2/3, width_1/2/3)
- 7 process parameters
- Environmental conditions

**Using your own data:**

To use a different dataset:
1. Prepare a CSV file with the same column structure as `cleaned_df.csv`:
   ```
   height_1, height_2, height_3, width_1, width_2, width_3,
   temp, humidity, layer_count, slicer_layer_height, slicer_layer_width,
   slicer_nozzle_speed, slicer_extrusion_multiplier
   ```
2. Replace `cleaned_df.csv` with your file
3. Delete `data/robocasting.db` (if it exists)
4. Run the application—it will auto-import your data

**Note:** The database schema supports additional metadata columns (`id`, `suggestion_id`, `archived`) which are added automatically. These are not required in your CSV.

---

## Contributing

Contributions to improve the application are welcome! We follow standard GitHub workflow:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/improvement-name`
3. **Make your changes**: Follow existing code style and add docstrings
4. **Test locally**: Ensure the app runs without errors
5. **Commit**: `git commit -am 'Add feature: description'`
6. **Push**: `git push origin feature/improvement-name`
7. **Submit a Pull Request**: Describe your changes and their motivation

**Types of contributions we're particularly interested in:**
- Additional optimization objectives (e.g., minimize material usage)
- Alternative kernels or ML models
- Improved visualization and plotting
- Export functionality (Excel, formatted reports)
- Unit tests and integration tests
- Documentation improvements
- Bug fixes

**Code style:**
- Follow PEP 8 for Python code
- Add docstrings to all functions (Google style)
- Keep functions focused and under 50 lines when possible
- Use type hints where appropriate

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

**Summary:** You are free to use, modify, and distribute this software for any purpose, including commercial use, as long as you include the original copyright notice and license.

---

## Contact

**Nazarii Mediukh**  
Email: n.mediukh@ipms.kyiv.ua  
Institution: Institute for Problems of Materials Science, National Academy of Sciences of Ukraine

**GitHub Issues:** For bug reports, feature requests, or technical questions, please [open an issue](https://github.com/matsimlab/robocasting-bayes-ui/issues).

---

## Acknowledgments

- **Institution:** Institute for Problems of Materials Science, National Academy of Sciences of Ukraine
- **Funding:** The work was supported by the Ukrainian Ministry of Education and Science under project number M/19-2024