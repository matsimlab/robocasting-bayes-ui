import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import cook_estimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from database import add_suggested_experiment


def suggest_next_experiment(data, models, bounds=None, n_calls=20, previous_points=None, diversity_weight=0.0, 
                           store_suggestion=True, suggestion_type="single_point"):
    """
    Suggest the next experiment point using Bayesian optimization to minimize
    the difference between slicer dimensions and actual printed dimensions.

    Args:
        data: DataFrame containing the existing experiment data
        models: Tuple of (width_model, height_model) GPR models
        bounds: Dictionary of parameter bounds for optimization
        n_calls: Number of iterations for the optimization
        previous_points: List of previously suggested points to avoid
        diversity_weight: Weight for the diversity penalty (0.0 = no penalty)

    Returns:
        Dictionary containing the suggested parameters for the next experiment
    """
    width_model, height_model = models

    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'temp': (data['temp'].min(), data['temp'].max()),
            'humidity': (data['humidity'].min(), data['humidity'].max()),
            'layer_count': (data['layer_count'].min(), data['layer_count'].max()),
            'slicer_layer_height': (data['slicer_layer_height'].min(), data['slicer_layer_height'].max()),
            'slicer_layer_width': (data['slicer_layer_width'].min(), data['slicer_layer_width'].max()),
            'slicer_nozzle_speed': (data['slicer_nozzle_speed'].min(), data['slicer_nozzle_speed'].max()),
            'slicer_extrusion_multiplier': (
                data['slicer_extrusion_multiplier'].min(), data['slicer_extrusion_multiplier'].max())
        }

    # Adjust bounds to ensure min < max for each parameter
    for key, (min_val, max_val) in bounds.items():
        if min_val == max_val:
            # Add a small range around the single value (±5%)
            delta = max(0.05 * abs(min_val), 0.01)  # At least 0.01 difference
            bounds[key] = (min_val - delta, max_val + delta)

    # Define the search space for optimization
    space = [
        Real(bounds['temp'][0], bounds['temp'][1], name='temp'),
        Real(bounds['humidity'][0], bounds['humidity'][1], name='humidity'),
        Integer(int(bounds['layer_count'][0]), int(bounds['layer_count'][1]), name='layer_count'),
        Real(bounds['slicer_layer_height'][0], bounds['slicer_layer_height'][1], name='slicer_layer_height'),
        Real(bounds['slicer_layer_width'][0], bounds['slicer_layer_width'][1], name='slicer_layer_width'),
        Real(bounds['slicer_nozzle_speed'][0], bounds['slicer_nozzle_speed'][1], name='slicer_nozzle_speed'),
        Real(bounds['slicer_extrusion_multiplier'][0], bounds['slicer_extrusion_multiplier'][1],
             name='slicer_extrusion_multiplier')
    ]

    # Function to calculate normalized distances to previous points
    def diversity_penalty(params, prev_points, sigma=0.2):
        """Calculate penalty to encourage diversity from previous suggestions"""
        if not prev_points or diversity_weight <= 0:
            return 0.0

        # Normalize parameters based on bounds
        bounds_list = [
            bounds['temp'], bounds['humidity'], bounds['layer_count'],
            bounds['slicer_layer_height'], bounds['slicer_layer_width'], 
            bounds['slicer_nozzle_speed'], bounds['slicer_extrusion_multiplier']
        ]

        # Convert to normalized space [0,1]
        params_norm = []
        for i, p in enumerate(params):
            low, high = bounds_list[i]
            range_val = high - low
            if range_val > 0:
                params_norm.append((p - low) / range_val)
            else:
                params_norm.append(0.5)  # Default for zero-range

        # Calculate minimum distance to previous points
        min_distance = float('inf')
        for point in prev_points:
            point_array = [
                point['temp'], point['humidity'], point['layer_count'],
                point['slicer_layer_height'], point['slicer_layer_width'],
                point['slicer_nozzle_speed'], point['slicer_extrusion_multiplier']
            ]

            # Convert previous point to normalized space
            point_norm = []
            for i, p in enumerate(point_array):
                low, high = bounds_list[i]
                range_val = high - low
                if range_val > 0:
                    point_norm.append((p - low) / range_val)
                else:
                    point_norm.append(0.5)

            # Euclidean distance in normalized space
            dist = np.sqrt(np.sum([(a - b) ** 2 for a, b in zip(params_norm, point_norm)]))
            min_distance = min(min_distance, dist)

        # Return a penalty that decreases with distance (encourage diversity)
        return np.exp(-min_distance / sigma)

    # Function to minimize - we want to find points where the predicted dimensions
    # match the slicer settings as closely as possible
    def dimension_mismatch_objective(params):
        X = np.array([params])

        # Extract slicer settings for comparison
        slicer_height = params[3]  # slicer_layer_height (index shifted due to layer_count)
        slicer_width = params[4]  # slicer_layer_width (index shifted due to layer_count)

        # Get predictions
        width_pred, width_std = width_model.predict(X, return_std=True)
        height_pred, height_std = height_model.predict(X, return_std=True)

        # Calculate absolute differences between predicted and target dimensions
        height_diff = abs(height_pred[0] - slicer_height)
        width_diff = abs(width_pred[0] - slicer_width)

        # Combined error metric (sum of absolute differences)
        combined_error = height_diff + width_diff

        # Include a small uncertainty component to avoid very uncertain regions
        uncertainty_penalty = 0.2 * (width_std[0] + height_std[0])

        # Add diversity penalty to avoid suggesting similar points
        diversity_pen = diversity_weight * diversity_penalty(params, previous_points)

        # Return the objective to minimize (lower is better)
        return combined_error + uncertainty_penalty + diversity_pen

    # Run the optimization
    result = gp_minimize(
        dimension_mismatch_objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=False
    )

    # Extract the best point
    suggested_point = {
        'temp': result.x[0],
        'humidity': result.x[1],
        'layer_count': int(result.x[2]),  # Convert to int since it's discrete
        'slicer_layer_height': result.x[3],
        'slicer_layer_width': result.x[4],
        'slicer_nozzle_speed': result.x[5],
        'slicer_extrusion_multiplier': result.x[6]
    }

    # Get the predicted values and uncertainties at this point
    X = np.array([[
        suggested_point['temp'],
        suggested_point['humidity'],
        suggested_point['layer_count'],
        suggested_point['slicer_layer_height'],
        suggested_point['slicer_layer_width'],
        suggested_point['slicer_nozzle_speed'],
        suggested_point['slicer_extrusion_multiplier']
    ]])

    width_pred, width_std = width_model.predict(X, return_std=True)
    height_pred, height_std = height_model.predict(X, return_std=True)

    # Calculate the dimension mismatches
    height_mismatch = abs(height_pred[0] - suggested_point['slicer_layer_height'])
    width_mismatch = abs(width_pred[0] - suggested_point['slicer_layer_width'])
    total_mismatch = height_mismatch + width_mismatch

    # Add predictions to the result
    suggested_point.update({
        'predicted_width': float(width_pred[0]),
        'width_uncertainty': float(width_std[0] * 1.96),  # 95% CI
        'predicted_height': float(height_pred[0]),
        'height_uncertainty': float(height_std[0] * 1.96),  # 95% CI
        'height_mismatch': float(height_mismatch),
        'width_mismatch': float(width_mismatch),
        'total_mismatch': float(total_mismatch)
    })
    
    # Store suggestion in the database with dataset size if requested
    if store_suggestion:
        dataset_size = len(data)
        add_suggested_experiment(suggested_point, dataset_size, suggestion_type)

    return suggested_point


def suggest_design_space_exploration(data, models, bounds=None, n_points=5, n_calls=20):
    """
    Suggest multiple points to explore the design space using Bayesian optimization
    to minimize dimensional errors between slicer settings and print outcomes.

    Args:
        data: DataFrame containing the existing experiment data
        models: Tuple of (width_model, height_model) GPR models
        bounds: Dictionary of parameter bounds for optimization
        n_points: Number of points to suggest
        n_calls: Number of iterations for each optimization

    Returns:
        List of dictionaries containing suggested parameters
    """
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'temp': (data['temp'].min(), data['temp'].max()),
            'humidity': (data['humidity'].min(), data['humidity'].max()),
            'layer_count': (data['layer_count'].min(), data['layer_count'].max()),
            'slicer_layer_height': (data['slicer_layer_height'].min(), data['slicer_layer_height'].max()),
            'slicer_layer_width': (data['slicer_layer_width'].min(), data['slicer_layer_width'].max()),
            'slicer_nozzle_speed': (data['slicer_nozzle_speed'].min(), data['slicer_nozzle_speed'].max()),
            'slicer_extrusion_multiplier': (
                data['slicer_extrusion_multiplier'].min(), data['slicer_extrusion_multiplier'].max())
        }

    # Adjust bounds to ensure min < max for each parameter
    for key, (min_val, max_val) in bounds.items():
        if min_val == max_val:
            # Add a small range around the single value (±5%)
            delta = max(0.05 * abs(min_val), 0.01)  # At least 0.01 difference
            bounds[key] = (min_val - delta, max_val + delta)

    suggested_points = []

    # Loop through and generate diverse points
    for i in range(n_points):
        # Increase diversity weight with each iteration to ensure variety
        diversity_weight = 2.0 * i  # Scale up for later iterations

        # Get next point, avoiding similarity with previously suggested points
        next_point = suggest_next_experiment(
            data,
            models,
            bounds,
            n_calls=n_calls,
            previous_points=suggested_points,
            diversity_weight=diversity_weight,
            store_suggestion=True,  # Store each point
            suggestion_type="design_space_exploration"  # Properly label them
        )

        suggested_points.append(next_point)

    return suggested_points