import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import cook_estimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from database import add_suggested_experiment


def suggest_next_experiment(data, models, bounds=None, n_calls=100, previous_points=None, diversity_weight=0.0,
                           store_suggestion=True, suggestion_type="single_point", include_uncertainty=None,
                           target_height=None, target_width=None):
    """
    Suggest the next experiment point using Bayesian optimization to minimize
    the difference between target dimensions and actual printed dimensions.
    
    NOTE: Now uses only 3 features (matching robocasting):
    - slicer_nozzle_speed
    - slicer_extrusion_multiplier  
    - layer_count

    Args:
        data: DataFrame containing the existing experiment data
        models: Tuple of (width_model, height_model) GPR models
        bounds: Dictionary of parameter bounds for optimization
        n_calls: Number of iterations for the optimization
        previous_points: List of previously suggested points to avoid
        diversity_weight: Weight for the diversity penalty (0.0 = no penalty)
        store_suggestion: Whether to store the suggestion in database
        suggestion_type: Type of suggestion for database tracking
        include_uncertainty: Whether to include uncertainty penalty (None = auto-decide based on suggestion_type)
        target_height: Target height in mm (e.g., 1.2mm for 80% of 1.5mm slicer height)
        target_width: Target width in mm (e.g., 2.0mm to match slicer width)

    Returns:
        Dictionary containing the suggested parameters for the next experiment
    """
    width_model, height_model = models
    
    # Get the scaler and features from the model
    scaler = width_model._scaler
    features = width_model._features  # ['slicer_nozzle_speed', 'slicer_extrusion_multiplier', 'layer_count']
    
    # Auto-decide whether to include uncertainty penalty
    if include_uncertainty is None:
        # For single point optimization, focus purely on dimension matching
        # For design space exploration, include uncertainty for better exploration
        include_uncertainty = (suggestion_type == "design_space_exploration")

    # Set default bounds if not provided - ONLY FOR THE 3 FEATURES USED
    if bounds is None:
        bounds = {
            'slicer_nozzle_speed': (3.0, 22.0),  # Corrected range
            'slicer_extrusion_multiplier': (0.4, 0.8),  # Corrected range
            'layer_count': (1, 3),  # Corrected range
            # Keep these for calculating expected dimensions, but don't optimize over them
            'slicer_layer_height': (data['slicer_layer_height'].min(), data['slicer_layer_height'].max()),
            'slicer_layer_width': (data['slicer_layer_width'].min(), data['slicer_layer_width'].max()),
        }

    # Adjust bounds to ensure min < max for each parameter
    for key, (min_val, max_val) in bounds.items():
        if min_val == max_val:
            # Add a small range around the single value (±5%)
            delta = max(0.05 * abs(min_val), 0.01)  # At least 0.01 difference
            bounds[key] = (min_val - delta, max_val + delta)

    # Define the search space for optimization - ONLY 3 FEATURES
    space = [
        Real(bounds['slicer_nozzle_speed'][0], bounds['slicer_nozzle_speed'][1], name='slicer_nozzle_speed'),
        Real(bounds['slicer_extrusion_multiplier'][0], bounds['slicer_extrusion_multiplier'][1],
             name='slicer_extrusion_multiplier'),
        Integer(int(bounds['layer_count'][0]), int(bounds['layer_count'][1]), name='layer_count'),
    ]
    
    # For calculating expected dimensions, we need to also suggest layer_height and layer_width
    # Use the mean values from the data for these
    default_layer_height = data['slicer_layer_height'].mean()
    default_layer_width = data['slicer_layer_width'].mean()

    # Function to calculate normalized distances to previous points
    def diversity_penalty(params, prev_points, sigma=0.2):
        """Calculate penalty to encourage diversity from previous suggestions"""
        if not prev_points or diversity_weight <= 0:
            return 0.0

        # Normalize parameters based on bounds - ONLY 3 FEATURES
        bounds_list = [
            bounds['slicer_nozzle_speed'],
            bounds['slicer_extrusion_multiplier'],
            bounds['layer_count']
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
                point['slicer_nozzle_speed'],
                point['slicer_extrusion_multiplier'],
                point['layer_count']
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
    # match the expected total dimensions as closely as possible
    iteration_count = [0]  # Use list to make it mutable in nested function
    
    def dimension_mismatch_objective(params):
        iteration_count[0] += 1
        
        # params is [slicer_nozzle_speed, slicer_extrusion_multiplier, layer_count]
        slicer_nozzle_speed = params[0]
        slicer_extrusion_multiplier = params[1]
        layer_count = params[2]
        
        # Scale the features using the same scaler from training
        X_raw = np.array([[slicer_nozzle_speed, slicer_extrusion_multiplier, layer_count]])
        X = scaler.transform(X_raw)
        
        # Use target values directly (user-specified)
        # If not provided, calculate from default slicer settings
        if target_height is not None:
            expected_total_height = target_height * layer_count
        else:
            # Fallback: use default layer height
            expected_total_height = default_layer_height * layer_count
        
        if target_width is not None:
            expected_width = target_width
        else:
            # Fallback: use default layer width
            expected_width = default_layer_width

        # Get predictions
        width_pred, width_std = width_model.predict(X, return_std=True)
        height_pred, height_std = height_model.predict(X, return_std=True)

        # Calculate absolute differences between predicted and expected total dimensions
        height_diff = abs(height_pred[0] - expected_total_height)
        width_diff = abs(width_pred[0] - expected_width)

        # Combined error metric (sum of absolute differences)
        # Weight height more heavily since it seems harder to optimize
        height_weight = 1.5  # Increase if height is more important
        width_weight = 1.0
        combined_error = height_weight * height_diff + width_weight * width_diff

        # Include uncertainty component only if specified
        uncertainty_penalty = 0.0
        if include_uncertainty:
            uncertainty_penalty = 0.2 * (width_std[0] + height_std[0])

        # Add diversity penalty to avoid suggesting similar points
        diversity_pen = diversity_weight * diversity_penalty(params, previous_points)

        # Total objective
        total_objective = combined_error + uncertainty_penalty + diversity_pen
        
        # Log this iteration
        print(f"Iteration {iteration_count[0]:3d}: "
              f"speed={slicer_nozzle_speed:5.2f}, mult={slicer_extrusion_multiplier:.3f}, layers={layer_count}, "
              f"h_diff={height_diff:.4f}, w_diff={width_diff:.4f}, "
              f"combined={combined_error:.6f}, objective={total_objective:.6f}")
        
        # Return the objective to minimize (lower is better)
        return total_objective

    # Run the optimization with multiple random starts to avoid local minima
    print("\n=== Starting Bayesian Optimization with Multiple Restarts ===")
    print(f"Running {3} independent optimizations to find global minimum...\n")
    
    best_overall_result = None
    best_overall_objective = float('inf')
    
    for restart in range(3):
        print(f"\n--- Restart {restart + 1}/3 ---")
        iteration_count[0] = 0  # Reset counter
        
        result = gp_minimize(
            dimension_mismatch_objective,
            space,
            n_calls=n_calls // 3,  # Divide calls among restarts
            random_state=42 + restart,  # Different random seed each time
            verbose=False
        )
        
        if result.fun < best_overall_objective:
            best_overall_objective = result.fun
            best_overall_result = result
            print(f"✓ New best objective: {result.fun:.6f}")
        else:
            print(f"  Objective: {result.fun:.6f} (not better than {best_overall_objective:.6f})")
    
    result = best_overall_result
    print(f"\n=== Best solution across all restarts: objective={result.fun:.6f} ===")

    # Extract the best point - params are [slicer_nozzle_speed, slicer_extrusion_multiplier, layer_count]
    suggested_point = {
        'slicer_nozzle_speed': result.x[0],
        'slicer_extrusion_multiplier': result.x[1],
        'layer_count': int(result.x[2]),  # Convert to int since it's discrete
        # Add fixed values for parameters not optimized
        'temp': data['temp'].mean(),  # Use mean value
        'humidity': data['humidity'].mean(),  # Use mean value  
        'slicer_layer_height': default_layer_height,
        'slicer_layer_width': default_layer_width,
    }

    # Get the predicted values and uncertainties at this point
    X_raw = np.array([[
        suggested_point['slicer_nozzle_speed'],
        suggested_point['slicer_extrusion_multiplier'],
        suggested_point['layer_count']
    ]])
    X = scaler.transform(X_raw)

    width_pred, width_std = width_model.predict(X, return_std=True)
    height_pred, height_std = height_model.predict(X, return_std=True)

    # Calculate the dimension mismatches using target values (not slicer values)
    if target_height is not None:
        expected_total_height = target_height * suggested_point['layer_count']
    else:
        # Fallback: use slicer layer height
        expected_total_height = suggested_point['slicer_layer_height'] * suggested_point['layer_count']
    
    if target_width is not None:
        expected_width = target_width
    else:
        # Fallback: use slicer layer width
        expected_width = suggested_point['slicer_layer_width']
    
    height_mismatch = abs(height_pred[0] - expected_total_height)
    width_mismatch = abs(width_pred[0] - expected_width)
    total_mismatch = height_mismatch + width_mismatch

    # Add predictions to the result
    suggested_point.update({
        'predicted_width': float(width_pred[0]),
        'width_uncertainty': float(width_std[0] * 1.96),  # 95% CI
        'predicted_height': float(height_pred[0]),
        'height_uncertainty': float(height_std[0] * 1.96),  # 95% CI
        'height_mismatch': float(height_mismatch),
        'width_mismatch': float(width_mismatch),
        'total_mismatch': float(total_mismatch),
        # Store target values for database tracking
        'target_height': target_height,
        'target_width': target_width,
    })
    
    # Store suggestion in the database with dataset size if requested
    if store_suggestion:
        dataset_size = len(data)
        add_suggested_experiment(suggested_point, dataset_size, suggestion_type)

    return suggested_point


def suggest_design_space_exploration(data, models, bounds=None, n_points=5, n_calls=60,
                                    target_height=None, target_width=None):
    """
    Suggest multiple points to explore the design space using Bayesian optimization
    to minimize dimensional errors between target dimensions and print outcomes.

    Args:
        data: DataFrame containing the existing experiment data
        models: Tuple of (width_model, height_model) GPR models
        bounds: Dictionary of parameter bounds for optimization
        n_points: Number of points to suggest
        n_calls: Number of iterations for each optimization
        target_height: Target height in mm (e.g., 1.2mm)
        target_width: Target width in mm (e.g., 2.0mm)

    Returns:
        List of dictionaries containing suggested parameters
    """
    # Set default bounds if not provided - ONLY FOR THE 3 FEATURES USED
    if bounds is None:
        bounds = {
            'slicer_nozzle_speed': (3.0, 22.0),  # Corrected range
            'slicer_extrusion_multiplier': (0.4, 0.8),  # Corrected range  
            'layer_count': (1, 3),  # Corrected range
            # Keep these for calculating expected dimensions
            'slicer_layer_height': (data['slicer_layer_height'].min(), data['slicer_layer_height'].max()),
            'slicer_layer_width': (data['slicer_layer_width'].min(), data['slicer_layer_width'].max()),
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
            suggestion_type="design_space_exploration",  # Properly label them
            include_uncertainty=True,  # Keep uncertainty for exploration mode
            target_height=target_height,
            target_width=target_width
        )

        suggested_points.append(next_point)

    return suggested_points