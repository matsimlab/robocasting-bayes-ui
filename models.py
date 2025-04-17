import numpy as np
import streamlit as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from skopt.callbacks import VerboseCallback
import time


@st.cache_resource
def train_gpr_models(data):
    """Train GPR models for width and height prediction with Bayesian hyperparameter optimization"""
    if data.empty:
        return None, None

    # Make sure we use data without the ID column for training
    if 'id' in data.columns:
        data = data.drop('id', axis=1)

    # Calculate average height and width
    data['avg_height'] = (data['height_1'] + data['height_2'] + data['height_3']) / 3
    data['avg_width'] = (data['width_1'] + data['width_2'] + data['width_3']) / 3

    # Features (input parameters)
    X = data[['temp', 'humidity', 'slicer_layer_height',
              'slicer_layer_width', 'slicer_nozzle_speed',
              'slicer_extrusion_multiplier']].values

    # Targets (average width and height)
    y_width = data['avg_width'].values
    y_height = data['avg_height'].values

    # Create placeholders for progress reporting
    progress_placeholder = st.empty()
    status_text = st.empty()

    # Define base kernel
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0] * X.shape[1], length_scale_bounds=(0.1, 10.0)) + WhiteKernel(
        noise_level=0.1, noise_level_bounds=(1e-5, 1.0))

    # Set up a custom callback for progress reporting
    class ProgressCallback:
        def __init__(self, n_iter, model_name):
            self.n_iter = n_iter
            self.current_iter = 0
            self.model_name = model_name
            self.progress_bar = progress_placeholder.progress(0)
            self.status = status_text
            self.status.info(f"Optimizing {model_name}... (0/{n_iter})")
            self.best_score = None

        def __call__(self, res):
            self.current_iter += 1
            progress = self.current_iter / self.n_iter
            self.progress_bar.progress(progress)

            # Update best score if available
            if hasattr(res, 'fun') and res.fun is not None:
                self.best_score = -res.fun  # Negate because BayesSearchCV maximizes negative MSE
                score_info = f" (Best Score: {self.best_score:.4f})"
            else:
                score_info = ""

            self.status.info(f"Optimizing {self.model_name}... ({self.current_iter}/{self.n_iter}){score_info}")

            # Return False to continue the optimization
            return False

    # Display initial message
    status_text.info("Starting model optimization...")

    # Set number of iterations
    n_iter = 20

    # Train width model with progress updates
    width_callback = ProgressCallback(n_iter, "Width Model")
    width_model = optimize_gpr_model(X, y_width, kernel, n_iter, width_callback)

    # Train height model with progress updates
    height_callback = ProgressCallback(n_iter, "Height Model")
    height_model = optimize_gpr_model(X, y_height, kernel, n_iter, height_callback)

    # Clear progress display when done
    progress_placeholder.empty()
    status_text.success("Optimization complete! Models are ready for predictions.")

    # Give the user time to see the completion message
    time.sleep(2)
    status_text.empty()

    return width_model, height_model


def optimize_gpr_model(X, y, kernel, n_iter, callback):
    """Optimize GPR model hyperparameters using Bayesian optimization"""
    # Create base GPR model
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, normalize_y=True)

    # Define hyperparameter search space
    search_spaces = {
        'alpha': Real(1e-10, 1e-2, prior='log-uniform'),
        'normalize_y': Categorical([True, False]),
    }

    # Create Bayesian search CV object with 3-fold cross-validation
    bayes_search = BayesSearchCV(
        estimator=gpr,
        search_spaces=search_spaces,
        scoring='neg_mean_squared_error',
        cv=3,
        n_iter=n_iter,  # Number of optimization iterations
        n_jobs=-1,  # Use all available cores
        random_state=42,
        verbose=0
    )

    # Fit the model with the callback
    with_callback = bayes_search.fit(X, y, callback=[callback])

    # Get best parameters
    best_params = with_callback.best_params_

    # Create optimized model with best parameters
    optimized_gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=best_params['alpha'],
        normalize_y=best_params['normalize_y'],
        n_restarts_optimizer=10,
        random_state=42
    )

    # Fit the final model
    optimized_gpr.fit(X, y)

    # Store the best parameters in the model for later access
    optimized_gpr.best_params_ = best_params

    return optimized_gpr


def make_prediction(params, models):
    """Make width and height predictions with uncertainty"""
    width_model, height_model = models

    # Format input for model
    X = np.array([[
        params['temp'],
        params['humidity'],
        params['slicer_layer_height'],
        params['slicer_layer_width'],
        params['slicer_nozzle_speed'],
        params['slicer_extrusion_multiplier']
    ]])

    # Get predictions
    width_pred, width_std = width_model.predict(X, return_std=True)
    height_pred, height_std = height_model.predict(X, return_std=True)


    return {
        'width': float(width_pred[0]),  # Now predicting average width
        'width_uncertainty': float(width_std[0] * 1.96),  # 95% confidence interval
        'height': float(height_pred[0]),  # Now predicting average height
        'height_uncertainty': float(height_std[0] * 1.96),  # 95% confidence interval
        'width_model_kernel': str(width_model.kernel_),
        'height_model_kernel': str(height_model.kernel_),
        'width_model_alpha': float(width_model.alpha),
        'height_model_alpha': float(height_model.alpha),
        'width_model_normalize_y': bool(width_model.normalize_y),
        'height_model_normalize_y': bool(height_model.normalize_y)
    }