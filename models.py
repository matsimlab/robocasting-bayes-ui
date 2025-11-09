import numpy as np
import streamlit as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold


@st.cache_resource
def train_gpr_models(data):
    """
    Train fixed GPR models for width and height prediction with nested CV evaluation.
    Uses GPR_RBF+White configuration from robocasting (fixed hyperparameters, no tuning).
    Returns: (width_model, height_model, cv_metrics)
    """
    if data.empty:
        st.error("No data available for training")
        return None, None, None

    st.info(f"Starting training with {len(data)} rows")
    
    # Make sure we use data without the metadata columns for training
    metadata_columns = ['id', 'suggestion_id', 'archived']  # These are for tracking, not features
    data_for_training = data.drop(columns=[col for col in metadata_columns if col in data.columns])
    
    st.info(f"After dropping metadata columns: {len(data_for_training)} rows")
    
    # Handle missing layer_count values by setting them to 1 (default)
    if 'layer_count' in data_for_training.columns:
        null_layer_count = data_for_training['layer_count'].isnull().sum()
        if null_layer_count > 0:
            st.info(f"Found {null_layer_count} NULL layer_count values, setting to 1")
            data_for_training['layer_count'] = data_for_training['layer_count'].fillna(1)
    else:
        # If layer_count column doesn't exist, add it with default value
        st.info("layer_count column missing, adding with default value 1")
        data_for_training['layer_count'] = 1
    
    # Check for NaN values in each column before dropping (excluding metadata columns)
    nan_summary = data_for_training.isnull().sum()
    total_nans = nan_summary.sum()
    if total_nans > 0:
        st.warning(f"Found NaN values in training data:\n{nan_summary[nan_summary > 0]}")
    
    # Drop any rows with remaining NaN values in the training features
    data_before_drop = len(data_for_training)
    data_for_training = data_for_training.dropna()
    data_after_drop = len(data_for_training)
    
    if data_after_drop < data_before_drop:
        st.info(f"Dropped {data_before_drop - data_after_drop} rows with NaN values in training features")
    
    if data_for_training.empty:
        st.error("No valid data remaining after cleaning. Please check your data for missing values.")
        return None, None, None

    # Use the cleaned data for training
    data = data_for_training

    # Calculate average height and width
    data['avg_height'] = (data['height_1'] + data['height_2'] + data['height_3']) / 3
    data['avg_width'] = (data['width_1'] + data['width_2'] + data['width_3']) / 3

    # Features (input parameters) - MATCH ROBOCASTING EXACTLY
    # Only use 3 features to match robocasting analysis
    from sklearn.preprocessing import StandardScaler
    
    features = ['slicer_nozzle_speed', 'slicer_extrusion_multiplier', 'layer_count']
    X_raw = data[features].values
    
    # Scale features like robocasting does
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Targets (average width and height)
    y_width = data['avg_width'].values
    y_height = data['avg_height'].values
    
    st.info(f"Using {len(features)} features (matching robocasting): {features}")

    # Display training message
    st.info("Training GPR models with fixed hyperparameters (GPR_RBF+White configuration)...")
    
    # Define fixed GPR kernel - copied from robocasting
    # GPR_RBF+White: ConstantKernel * RBF + WhiteKernel
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    
    # Create GPR models with fixed hyperparameters (no tuning)
    # alpha=1e-10 for numerical stability with WhiteKernel
    width_model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42
    )
    
    height_model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42
    )
    
    # Perform nested cross-validation for evaluation
    st.info("Evaluating models using 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate width model
    width_mae_scores = -cross_val_score(
        width_model, X, y_width,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    width_r2_scores = cross_val_score(
        width_model, X, y_width,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )
    
    # Evaluate height model
    height_mae_scores = -cross_val_score(
        height_model, X, y_height,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    height_r2_scores = cross_val_score(
        height_model, X, y_height,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )
    
    # Store CV metrics
    cv_metrics = {
        'width': {
            'mae_mean': width_mae_scores.mean(),
            'mae_std': width_mae_scores.std(),
            'r2_mean': width_r2_scores.mean(),
            'r2_std': width_r2_scores.std()
        },
        'height': {
            'mae_mean': height_mae_scores.mean(),
            'mae_std': height_mae_scores.std(),
            'r2_mean': height_r2_scores.mean(),
            'r2_std': height_r2_scores.std()
        },
        'n_samples': len(X),
        'n_folds': 5
    }
    
    # Train final models on all data
    st.info("Training final models on full dataset...")
    width_model.fit(X, y_width)
    height_model.fit(X, y_height)
    
    st.success("✅ Model training and evaluation complete!")
    
    # Store scaler and feature list in the models for prediction
    width_model._scaler = scaler
    width_model._features = features
    height_model._scaler = scaler
    height_model._features = features
    
    return width_model, height_model, cv_metrics


def make_prediction(params, models):
    """Make width and height predictions with uncertainty"""
    width_model, height_model, cv_metrics = models

    # Ensure layer_count is present in params
    if 'layer_count' not in params:
        params['layer_count'] = 1  # Default value

    # Extract only the features that were used in training (3 features)
    features = width_model._features  # ['slicer_nozzle_speed', 'slicer_extrusion_multiplier', 'layer_count']
    scaler = width_model._scaler
    
    # Build feature vector in correct order
    X_raw = np.array([[
        params['slicer_nozzle_speed'],
        params['slicer_extrusion_multiplier'],
        params['layer_count']
    ]])
    
    # Scale features using the same scaler from training
    X = scaler.transform(X_raw)

    # Get predictions
    width_pred, width_std = width_model.predict(X, return_std=True)
    height_pred, height_std = height_model.predict(X, return_std=True)

    return {
        'width': float(width_pred[0]),
        'width_uncertainty': float(width_std[0] * 1.96),  # 95% confidence interval
        'height': float(height_pred[0]),
        'height_uncertainty': float(height_std[0] * 1.96),  # 95% confidence interval
        'width_model_kernel': str(width_model.kernel_),
        'height_model_kernel': str(height_model.kernel_),
        'width_model_alpha': float(width_model.alpha),
        'height_model_alpha': float(height_model.alpha),
        'width_model_normalize_y': bool(width_model.normalize_y),
        'height_model_normalize_y': bool(height_model.normalize_y),
        # Include CV metrics in prediction results
        'cv_metrics': cv_metrics
    }
