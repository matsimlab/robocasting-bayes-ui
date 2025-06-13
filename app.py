import streamlit as st
import pandas as pd
import numpy as np

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Robocasting Experiments",
    page_icon="🤖",
    layout="wide"
)

# Import modules
from database import init_db, get_data, get_data_for_display, add_data_point, delete_data_point, \
    delete_multiple_data_points, get_suggested_experiments, get_suggested_experiments_for_dropdown, get_suggestion_by_id
from models import train_gpr_models, make_prediction
from visualization import create_scatter_plot, create_summary_stats
from utils import add_custom_css, validate_new_data_point
from auth import login_page, show_logout_button, show_user_management
from next_point import suggest_next_experiment, suggest_design_space_exploration

# Add custom CSS
add_custom_css()

# Initialize the database
init_db()

# Initialize session state for app functionality
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'sidebar_page' not in st.session_state:
    st.session_state.sidebar_page = "Data Explorer"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "next_experiment_point" not in st.session_state:
    st.session_state.next_experiment_point = None
if "selected_suggestion_id" not in st.session_state:
    st.session_state.selected_suggestion_id = None
if "prefilled_from_suggestion" not in st.session_state:
    st.session_state.prefilled_from_suggestion = False

# Show the logout button in the sidebar
show_logout_button()

# Check authentication
if not login_page():
    st.stop()


# Function to store prediction parameters for add_data form
def store_params_for_form():
    st.session_state.stored_temp = st.session_state.temp_input
    st.session_state.stored_humidity = st.session_state.humidity_input
    st.session_state.stored_layer_count = st.session_state.layer_count_input
    st.session_state.stored_layer_height = st.session_state.layer_height_input
    st.session_state.stored_layer_width = st.session_state.layer_width_input
    st.session_state.stored_nozzle_speed = st.session_state.nozzle_speed_input
    st.session_state.stored_extrusion_multiplier = st.session_state.extrusion_multiplier_input
    st.session_state.show_add_data = True
    st.session_state.prefilled_from_prediction = True  # Flag to indicate source
    # Update sidebar selection
    st.session_state.sidebar_page = "Add New Data"


# Function to store parameters from next experiment suggestions with suggestion ID
def store_suggestion_params_for_form(suggestion_id):
    next_point = st.session_state.next_experiment_point
    st.session_state.stored_temp = next_point['temp']
    st.session_state.stored_humidity = next_point['humidity']
    st.session_state.stored_layer_count = next_point['layer_count']
    st.session_state.stored_layer_height = next_point['slicer_layer_height']
    st.session_state.stored_layer_width = next_point['slicer_layer_width']
    st.session_state.stored_nozzle_speed = next_point['slicer_nozzle_speed']
    st.session_state.stored_extrusion_multiplier = next_point['slicer_extrusion_multiplier']
    st.session_state.selected_suggestion_id = suggestion_id
    st.session_state.prefilled_from_suggestion = True
    st.session_state.show_add_data = True
    st.session_state.sidebar_page = "Add New Data"


# Function to store parameters from optimization history with suggestion ID  
def store_history_suggestion_params_for_form(suggestion_data, suggestion_id):
    st.session_state.stored_temp = suggestion_data['temp']
    st.session_state.stored_humidity = suggestion_data['humidity']
    st.session_state.stored_layer_count = suggestion_data['layer_count']
    st.session_state.stored_layer_height = suggestion_data['slicer_layer_height']
    st.session_state.stored_layer_width = suggestion_data['slicer_layer_width']
    st.session_state.stored_nozzle_speed = suggestion_data['slicer_nozzle_speed']
    st.session_state.stored_extrusion_multiplier = suggestion_data['slicer_extrusion_multiplier']
    st.session_state.selected_suggestion_id = suggestion_id
    st.session_state.prefilled_from_suggestion = True
    st.session_state.show_add_data = True
    st.session_state.sidebar_page = "Add New Data"


# Add sidebar navigation - use the saved state for the default value
page = st.sidebar.radio("Navigation", ["Data Explorer", "Predictions", "Next Experiment", "Add New Data", "Optimization History"],
                        index=["Data Explorer", "Predictions", "Next Experiment", "Add New Data", "Optimization History"].index(
                            st.session_state.sidebar_page) if st.session_state.sidebar_page in 
                            ["Data Explorer", "Predictions", "Next Experiment", "Add New Data", "Optimization History"] else 0,
                        key="navigation")

# Add user management to bottom of sidebar
show_user_management()

# Handle page changes
if page != st.session_state.sidebar_page:
    st.session_state.sidebar_page = page
    st.rerun()

# Override page selection if we need to show Add New Data
if 'show_add_data' in st.session_state and st.session_state.show_add_data:
    page = "Add New Data"
    # Reset flag after navigation but keep sidebar visible
    st.session_state.show_add_data = False
    # Update the sidebar_page to match current page
    if st.session_state.sidebar_page != page:
        st.session_state.sidebar_page = page
        st.rerun()

# Display content based on selected tab
if page == "Data Explorer":
    st.header("Dataset")
    
    # Add a button to clear cache and reload data if there are issues
    if st.button("🔄 Clear Cache & Reload Data", help="Use this if you're experiencing data issues"):
        st.cache_resource.clear()
        st.rerun()

    # Load data
    full_data, display_data = get_data_for_display()

    # Handle data and deletion
    if not full_data.empty:
        st.header("Dataset")

        # Display data table without the ID column
        st.dataframe(display_data, use_container_width=True)

        # Add delete functionality with a more compatible approach
        with st.expander("Delete Data Points"):
            st.warning("⚠️ Select a row to delete. This action cannot be undone.")

            # Simple numeric selector for row ID - but show more meaningful info
            id_to_info = {}
            for idx, row in full_data.iterrows():
                row_id = row['id']
                # Create a descriptive label for each row
                info = f"Row {idx + 1}: Temp={row['temp']}°C, Width={row['width_1']:.2f}mm, Height={row['height_1']:.2f}mm"
                id_to_info[row_id] = info

            selected_id = st.selectbox(
                "Select row to delete:",
                options=list(id_to_info.keys()),
                format_func=lambda x: id_to_info[x]
            )

            if st.button("Delete Selected Row", type="primary"):
                if delete_data_point(selected_id):
                    st.success(f"Successfully deleted row")
                    # Clear the prediction if it exists, as the data has changed
                    if 'prediction' in st.session_state:
                        st.session_state.prediction = None
                    # Rerun the app to refresh the data
                    st.rerun()
                else:
                    st.error(f"Failed to delete row")
    else:
        st.info("No data available. Add data points to see the dataset.")

    # Display data summary
    st.header("Data Summary")
    if not full_data.empty:
        stats = create_summary_stats(full_data)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", stats['count'])
        with col2:
            st.metric("Avg Width", f"{stats['avg_width']:.2f} mm")
        with col3:
            st.metric("Avg Height", f"{stats['avg_height']:.2f} mm")
        with col4:
            st.metric("Avg Temperature", f"{stats['avg_temp']:.1f} °C")

        # Visualization section
        st.header("Data Visualization")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Width vs Temperature scatter plot
            st.plotly_chart(
                create_scatter_plot(full_data, "temp", "width_1", "Width vs Temperature"),
                use_container_width=True
            )

        with viz_col2:
            # Height vs Temperature scatter plot
            st.plotly_chart(
                create_scatter_plot(full_data, "temp", "height_1", "Height vs Temperature"),
                use_container_width=True
            )

        # Additional visualizations
        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            # Width vs Extrusion Multiplier
            st.plotly_chart(
                create_scatter_plot(
                    full_data,
                    "slicer_extrusion_multiplier",
                    "width_1",
                    "Width vs Extrusion Multiplier"
                ),
                use_container_width=True
            )

        with viz_col4:
            # Height vs Extrusion Multiplier
            st.plotly_chart(
                create_scatter_plot(
                    full_data,
                    "slicer_extrusion_multiplier",
                    "height_1",
                    "Height vs Extrusion Multiplier"
                ),
                use_container_width=True
            )

elif page == "Predictions":
    st.header("Gaussian Process Regression Predictions")
    
    # Add a button to clear cache and reload data if there are issues
    if st.button("🔄 Clear Model Cache & Reload", help="Use this if you're experiencing model training issues"):
        st.cache_resource.clear()
        st.rerun()

    # Load data - use the full data with IDs for model training
    data = get_data()

    if not data.empty:
        st.success(f"Loaded {len(data)} records for training")
        
        # Display data summary
        st.write("**Data Overview:**")
        st.write(f"- Total records: {len(data)}")
        st.write(f"- Columns: {list(data.columns)}")
        
        # Check for required columns
        required_cols = ['temp', 'humidity', 'slicer_layer_height', 'slicer_layer_width', 
                        'slicer_nozzle_speed', 'slicer_extrusion_multiplier',
                        'height_1', 'height_2', 'height_3', 'width_1', 'width_2', 'width_3']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
        
        # Check layer_count column
        if 'layer_count' not in data.columns:
            st.warning("layer_count column missing, will use default value 1")
        else:
            layer_counts = data['layer_count'].value_counts().sort_index()
            st.write(f"**Layer count distribution:** {dict(layer_counts)}")
        
        try:
            # Train models
            width_model, height_model = train_gpr_models(data)
            
            if width_model is None or height_model is None:
                st.error("Failed to train models. Please check the data and try clearing the cache.")
                st.stop()
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            st.write("Try clearing the cache and reloading the data.")
            st.stop()

        # Prediction form
        st.subheader("Enter Parameters")

        col1, col2 = st.columns(2)

        with col1:
            temp = st.number_input("Temperature (°C)",
                                   min_value=float(data['temp'].min() - 1),
                                   max_value=float(data['temp'].max() + 1),
                                   value=float(data['temp'].mean()),
                                   step=0.1,
                                   format="%.1f",
                                   key="temp_input")

            humidity = st.number_input("Humidity (%)",
                                       min_value=float(data['humidity'].min() - 5),
                                       max_value=float(data['humidity'].max() + 5),
                                       value=float(data['humidity'].mean()),
                                       step=1.0,
                                       format="%.1f",
                                       key="humidity_input")

            layer_count = st.number_input("Layer Count",
                                         min_value=int(data['layer_count'].min()),
                                         max_value=int(data['layer_count'].max()),
                                         value=int(data['layer_count'].mean()),
                                         step=1,
                                         format="%d",
                                         key="layer_count_input")

            slicer_layer_height = st.number_input("Slicer Layer Height",
                                                  min_value=float(data['slicer_layer_height'].min() - 0.1),
                                                  max_value=float(data['slicer_layer_height'].max() + 0.1),
                                                  value=float(data['slicer_layer_height'].mean()),
                                                  step=0.1,
                                                  format="%.1f",
                                                  key="layer_height_input")

        with col2:
            slicer_layer_width = st.number_input("Slicer Layer Width",
                                                 min_value=float(data['slicer_layer_width'].min() - 0.1),
                                                 max_value=float(data['slicer_layer_width'].max() + 0.1),
                                                 value=float(data['slicer_layer_width'].mean()),
                                                 step=0.1,
                                                 format="%.1f",
                                                 key="layer_width_input")

            slicer_nozzle_speed = st.number_input("Slicer Nozzle Speed",
                                                  min_value=float(data['slicer_nozzle_speed'].min() - 1),
                                                  max_value=float(data['slicer_nozzle_speed'].max() + 1),
                                                  value=float(data['slicer_nozzle_speed'].mean()),
                                                  step=1.0,
                                                  format="%.1f",
                                                  key="nozzle_speed_input")

            slicer_extrusion_multiplier = st.number_input("Extrusion Multiplier",
                                                          min_value=float(
                                                              data['slicer_extrusion_multiplier'].min() - 0.1),
                                                          max_value=float(
                                                              data['slicer_extrusion_multiplier'].max() + 0.1),
                                                          value=float(data['slicer_extrusion_multiplier'].mean()),
                                                          step=0.01,
                                                          format="%.2f",
                                                          key="extrusion_multiplier_input")

        # Parameters for prediction
        params = {
            'temp': temp,
            'humidity': humidity,
            'layer_count': layer_count,
            'slicer_layer_height': slicer_layer_height,
            'slicer_layer_width': slicer_layer_width,
            'slicer_nozzle_speed': slicer_nozzle_speed,
            'slicer_extrusion_multiplier': slicer_extrusion_multiplier
        }

        # Button to make prediction
        if st.button("Calculate Prediction"):
            # Make prediction
            prediction = make_prediction(params, (width_model, height_model))

            # Store prediction in session state
            st.session_state.prediction = prediction

            # Display prediction results
            st.subheader("Prediction Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Predicted Avg Width", f"{prediction['width']:.2f} mm")
                st.metric("Width Uncertainty", f"±{prediction['width_uncertainty']:.3f} mm")

                # Create a simple visualization with confidence interval
                width_range = (prediction['width'] - prediction['width_uncertainty'],
                               prediction['width'] + prediction['width_uncertainty'])
                st.markdown("**95% Confidence Interval:**")
                # Use a normalized progress bar (clamp to 0-1 range)
                normalized_width = min(1.0, max(0.0, prediction['width'] / 10.0))  # Normalize to reasonable range
                st.progress(normalized_width)
                st.markdown(f"**Range:** {width_range[0]:.2f} mm to {width_range[1]:.2f} mm")

            with result_col2:
                st.metric("Predicted Avg Height", f"{prediction['height']:.2f} mm")
                st.metric("Height Uncertainty", f"±{prediction['height_uncertainty']:.3f} mm")

                # Create a simple visualization with confidence interval
                height_range = (prediction['height'] - prediction['height_uncertainty'],
                                prediction['height'] + prediction['height_uncertainty'])
                st.markdown("**95% Confidence Interval:**")
                # Use a normalized progress bar (clamp to 0-1 range)
                normalized_height = min(1.0, max(0.0, prediction['height'] / 5.0))  # Normalize to reasonable range
                st.progress(normalized_height)
                st.markdown(f"**Range:** {height_range[0]:.2f} mm to {height_range[1]:.2f} mm")

            # Display model information
            with st.expander("Model Details"):
                st.subheader("Width Model")
                st.markdown("**Kernel:**")
                st.code(prediction['width_model_kernel'])
                st.markdown("**Hyperparameters:**")
                st.code(
                    f"alpha: {prediction['width_model_alpha']:.10f} (noise level)\nnormalize_y: {prediction['width_model_normalize_y']}")

                st.subheader("Height Model")
                st.markdown("**Kernel:**")
                st.code(prediction['height_model_kernel'])
                st.markdown("**Hyperparameters:**")
                st.code(
                    f"alpha: {prediction['height_model_alpha']:.10f} (noise level)\nnormalize_y: {prediction['height_model_normalize_y']}")

            # Add option to transfer these parameters to the "Add New Data" tab
            st.subheader("Use these parameters for a new experiment")
            transfer_button = st.button("Transfer to Add New Data", on_click=store_params_for_form)

        # Display previous prediction if available and button wasn't pressed
        elif st.session_state.prediction is not None:
            prediction = st.session_state.prediction

            st.subheader("Previous Prediction Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Predicted Avg Width", f"{prediction['width']:.2f} mm")
                st.metric("Width Uncertainty", f"±{prediction['width_uncertainty']:.3f} mm")

                # Create a simple visualization with confidence interval
                width_range = (prediction['width'] - prediction['width_uncertainty'],
                               prediction['width'] + prediction['width_uncertainty'])
                st.markdown("**95% Confidence Interval:**")
                # Use a normalized progress bar (clamp to 0-1 range)
                normalized_width = min(1.0, max(0.0, prediction['width'] / 10.0))  # Normalize to reasonable range
                st.progress(normalized_width)
                st.markdown(f"**Range:** {width_range[0]:.2f} mm to {width_range[1]:.2f} mm")

            with result_col2:
                st.metric("Predicted Avg Height", f"{prediction['height']:.2f} mm")
                st.metric("Height Uncertainty", f"±{prediction['height_uncertainty']:.3f} mm")

                # Create a simple visualization with confidence interval
                height_range = (prediction['height'] - prediction['height_uncertainty'],
                                prediction['height'] + prediction['height_uncertainty'])
                st.markdown("**95% Confidence Interval:**")
                # Use a normalized progress bar (clamp to 0-1 range)
                normalized_height = min(1.0, max(0.0, prediction['height'] / 5.0))  # Normalize to reasonable range
                st.progress(normalized_height)
                st.markdown(f"**Range:** {height_range[0]:.2f} mm to {height_range[1]:.2f} mm")

            # Display model information
            with st.expander("Model Details"):
                st.subheader("Width Model")
                st.markdown("**Kernel:**")
                st.code(prediction['width_model_kernel'])
                st.markdown("**Hyperparameters:**")
                st.code(
                    f"alpha: {prediction['width_model_alpha']:.10f} (noise level)\nnormalize_y: {prediction['width_model_normalize_y']}")

                st.subheader("Height Model")
                st.markdown("**Kernel:**")
                st.code(prediction['height_model_kernel'])
                st.markdown("**Hyperparameters:**")
                st.code(
                    f"alpha: {prediction['height_model_alpha']:.10f} (noise level)\nnormalize_y: {prediction['height_model_normalize_y']}")

            # Add option to transfer these parameters to the "Add New Data" tab
            st.subheader("Use these parameters for a new experiment")
            transfer_button = st.button("Transfer to Add New Data", on_click=store_params_for_form)
    else:
        st.info("No data available. Add data points to make predictions.")

elif page == "Next Experiment":
    st.header("Next Experiment Suggestion")

    # Load data
    data = get_data()

    if not data.empty:
        # Train models if we haven't already
        width_model, height_model = train_gpr_models(data)

        # Explanation
        st.markdown("""
        This tab uses Bayesian optimization to suggest the next experiment point that will 
        help minimize the difference between your slicer settings and the actual printed dimensions.

        The algorithm looks for parameter combinations where the predicted height and width 
        output will most closely match the height and width you specify in your slicer settings.
        """)

        # Options for suggestion
        st.subheader("Suggestion Options")

        suggestion_type = st.radio(
            "Suggestion approach:",
            ["Optimal Dimension Matching (single point)", "Design Space Exploration (multiple points)"]
        )

        if suggestion_type == "Optimal Dimension Matching (single point)":
            # Button to generate suggestion
            if st.button("Suggest Next Experiment"):
                with st.spinner("Finding optimal parameters for dimension matching..."):
                    next_point = suggest_next_experiment(data, (width_model, height_model), 
                                                        store_suggestion=True, 
                                                        suggestion_type="single_point")
                    st.session_state.next_experiment_point = next_point
        else:
            # Multiple point suggestion
            num_points = st.slider("Number of suggestions:", min_value=2, max_value=10, value=5)

            if st.button("Generate Suggestions"):
                with st.spinner(f"Finding {num_points} diverse experiment points..."):
                    next_points = suggest_design_space_exploration(
                        data, (width_model, height_model), n_points=num_points
                    )
                    st.session_state.next_experiment_points = next_points

        # Display suggestion if available
        if suggestion_type == "Optimal Dimension Matching (single point)" and st.session_state.next_experiment_point is not None:
            st.subheader("Suggested Experiment Parameters")

            next_point = st.session_state.next_experiment_point

            # Create columns for parameter display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temperature", f"{next_point['temp']:.1f} °C")
                st.metric("Humidity", f"{next_point['humidity']:.2f}")
            with col2:
                st.metric("Layer Count", f"{next_point['layer_count']}")
                st.metric("Layer Height", f"{next_point['slicer_layer_height']:.1f}")
            with col3:
                st.metric("Layer Width", f"{next_point['slicer_layer_width']:.1f}")
                st.metric("Nozzle Speed", f"{next_point['slicer_nozzle_speed']:.1f}")
                st.metric("Extrusion Multiplier", f"{next_point['slicer_extrusion_multiplier']:.2f}")

            # Display prediction for this point
            st.subheader("Expected Outcome")
            pred_col1, pred_col2 = st.columns(2)

            with pred_col1:
                st.metric("Predicted Width", f"{next_point['predicted_width']:.2f} mm")
                st.metric("Width Mismatch", f"{next_point['width_mismatch']:.3f} mm")

            with pred_col2:
                st.metric("Predicted Height", f"{next_point['predicted_height']:.2f} mm")
                st.metric("Height Mismatch", f"{next_point['height_mismatch']:.3f} mm")

            st.info(f"Total dimension mismatch: {next_point['total_mismatch']:.3f} mm")

            # Add visual comparison of target vs predicted dimensions
            st.subheader("Dimension Comparison")

            # Calculate percentage match for height and width
            height_match = 100 * (1 - next_point['height_mismatch'] / next_point['slicer_layer_height'])
            width_match = 100 * (1 - next_point['width_mismatch'] / next_point['slicer_layer_width'])

            # Ensure percentages are within reasonable bounds
            height_match = max(0, min(100, height_match))
            width_match = max(0, min(100, width_match))

            # Display match percentages
            match_col1, match_col2 = st.columns(2)
            with match_col1:
                st.metric("Height Match", f"{height_match:.1f}%")
                st.text(f"Target: {next_point['slicer_layer_height']:.2f} mm")
                st.text(f"Expected: {next_point['predicted_height']:.2f} mm")
                st.progress(height_match / 100)

            with match_col2:
                st.metric("Width Match", f"{width_match:.1f}%")
                st.text(f"Target: {next_point['slicer_layer_width']:.2f} mm")
                st.text(f"Expected: {next_point['predicted_width']:.2f} mm")
                st.progress(width_match / 100)

            # Add button to use these parameters for a new experiment
            if st.button("Use These Parameters"):
                # Find the suggestion ID for this point (most recent suggestion)
                suggested_df = get_suggested_experiments(limit=1)
                suggestion_id = suggested_df.iloc[0]['id'] if not suggested_df.empty else None
                
                # Store parameters in session state with suggestion ID
                store_suggestion_params_for_form(suggestion_id)
                st.rerun()

        elif suggestion_type == "Design Space Exploration (multiple points)" and hasattr(st.session_state,
                                                                                         'next_experiment_points'):
            st.subheader("Suggested Experiments")

            # Create tabs for each suggestion
            suggestion_tabs = st.tabs(
                [f"Suggestion {i + 1}" for i in range(len(st.session_state.next_experiment_points))])

            for i, (tab, point) in enumerate(zip(suggestion_tabs, st.session_state.next_experiment_points)):
                with tab:
                    # Parameters
                    param_col1, param_col2, param_col3 = st.columns(3)
                    with param_col1:
                        st.metric("Temperature", f"{point['temp']:.1f} °C")
                        st.metric("Humidity", f"{point['humidity']:.2f}")
                    with param_col2:
                        st.metric("Layer Count", f"{point['layer_count']}")
                        st.metric("Layer Height", f"{point['slicer_layer_height']:.1f}")
                    with param_col3:
                        st.metric("Layer Width", f"{point['slicer_layer_width']:.1f}")
                        st.metric("Nozzle Speed", f"{point['slicer_nozzle_speed']:.1f}")
                        st.metric("Extrusion Multiplier", f"{point['slicer_extrusion_multiplier']:.2f}")

                    # Predictions
                    st.divider()
                    pred_col1, pred_col2 = st.columns(2)

                    with pred_col1:
                        st.metric("Predicted Width", f"{point['predicted_width']:.2f} mm")
                        st.metric("Width Mismatch", f"{point['width_mismatch']:.3f} mm")

                    with pred_col2:
                        st.metric("Predicted Height", f"{point['predicted_height']:.2f} mm")
                        st.metric("Height Mismatch", f"{point['height_mismatch']:.3f} mm")

                    st.info(f"Total dimension mismatch: {point['total_mismatch']:.3f} mm")

                    # Calculate percentage match for height and width
                    height_match = 100 * (1 - point['height_mismatch'] / point['slicer_layer_height'])
                    width_match = 100 * (1 - point['width_mismatch'] / point['slicer_layer_width'])

                    # Ensure percentages are within reasonable bounds
                    height_match = max(0, min(100, height_match))
                    width_match = max(0, min(100, width_match))

                    # Display match percentages
                    match_col1, match_col2 = st.columns(2)
                    with match_col1:
                        st.metric("Height Match", f"{height_match:.1f}%")
                        st.text(f"Target: {point['slicer_layer_height']:.2f} mm")
                        st.text(f"Expected: {point['predicted_height']:.2f} mm")
                        st.progress(height_match / 100)

                    with match_col2:
                        st.metric("Width Match", f"{width_match:.1f}%")
                        st.text(f"Target: {point['slicer_layer_width']:.2f} mm")
                        st.text(f"Expected: {point['predicted_width']:.2f} mm")
                        st.progress(width_match / 100)

                    # Use parameters button
                    if st.button(f"Use Parameters {i + 1}", key=f"use_params_{i}"):
                        # Find the suggestion ID for this point (get recent suggestions and find matching one)
                        suggested_df = get_suggested_experiments(limit=len(st.session_state.next_experiment_points))
                        # Match based on parameters (since they might not be in exact order)
                        suggestion_id = None
                        if not suggested_df.empty:
                            for _, row in suggested_df.iterrows():
                                if (abs(row['temp'] - point['temp']) < 0.01 and 
                                    abs(row['humidity'] - point['humidity']) < 0.001 and
                                    abs(row.get('layer_count', 1) - point.get('layer_count', 1)) < 0.1 and
                                    abs(row['slicer_layer_height'] - point['slicer_layer_height']) < 0.01):
                                    suggestion_id = row['id']
                                    break
                        
                        # Store parameters in session state with suggestion ID
                        store_history_suggestion_params_for_form(point, suggestion_id)
                        st.rerun()
    else:
        st.info("No data available. Add data points to generate experiment suggestions.")

elif page == "Add New Data":
    st.header("Add New Data Point")
    
    # Show notification if parameters were transferred from another page
    if st.session_state.get('prefilled_from_prediction', False):
        st.success("✨ Parameters transferred from Predictions page")
        # Clear the flag after showing the message
        st.session_state.prefilled_from_prediction = False
    elif st.session_state.get('prefilled_from_suggestion', False):
        st.success("✨ Parameters transferred from suggestions")
        # Don't clear this flag as it's used elsewhere
    
    # Debug info (can be removed later)
    debug_enabled = st.checkbox("Show debug info", value=False, key="debug_checkbox")
    if debug_enabled:
        st.session_state.show_debug_info = True
        st.write("**Stored values in session state:**")
        st.write(f"- Temp: {st.session_state.get('stored_temp', 'Not set')}")
        st.write(f"- Humidity: {st.session_state.get('stored_humidity', 'Not set')}")
        st.write(f"- Layer Count: {st.session_state.get('stored_layer_count', 'Not set')}")
        st.write(f"- Layer Height: {st.session_state.get('stored_layer_height', 'Not set')}")
        st.write(f"- Layer Width: {st.session_state.get('stored_layer_width', 'Not set')}")
        st.write(f"- Nozzle Speed: {st.session_state.get('stored_nozzle_speed', 'Not set')}")
        st.write(f"- Extrusion Multiplier: {st.session_state.get('stored_extrusion_multiplier', 'Not set')}")
        
        # Manual clear button for testing
        if st.button("Clear stored parameters"):
            stored_params_to_clear = [
                'stored_temp', 'stored_humidity', 'stored_layer_count',
                'stored_layer_height', 'stored_layer_width', 'stored_nozzle_speed', 
                'stored_extrusion_multiplier', 'selected_suggestion_id', 
                'prefilled_from_suggestion', 'prefilled_from_prediction'
            ]
            for param in stored_params_to_clear:
                if param in st.session_state:
                    del st.session_state[param]
            st.rerun()
    else:
        st.session_state.show_debug_info = False

    # Get available suggestions for dropdown
    suggestions = get_suggested_experiments_for_dropdown()
    
    # Use parameters from stored values if available
    default_temp = st.session_state.get('stored_temp', 20.0)
    default_humidity = st.session_state.get('stored_humidity', 50.0)  # Default to 50%
    default_layer_count = st.session_state.get('stored_layer_count', 1)
    default_layer_height = st.session_state.get('stored_layer_height', 0.8)
    default_layer_width = st.session_state.get('stored_layer_width', 1.5)
    default_nozzle_speed = st.session_state.get('stored_nozzle_speed', 8.0)
    default_extrusion_multiplier = st.session_state.get('stored_extrusion_multiplier', 1.0)
    
    # Show calculated defaults in debug mode
    if st.session_state.get('show_debug_info', False):
        st.write("**Calculated default values:**")
        st.write(f"- Default Temp: {default_temp}")
        st.write(f"- Default Humidity: {default_humidity}")
        st.write(f"- Default Layer Count: {default_layer_count}")
        st.write(f"- Default Layer Height: {default_layer_height}")
        st.write(f"- Default Layer Width: {default_layer_width}")
        st.write(f"- Default Nozzle Speed: {default_nozzle_speed}")
        st.write(f"- Default Extrusion Multiplier: {default_extrusion_multiplier}")
    
    # Determine default suggestion selection
    default_suggestion = st.session_state.get('selected_suggestion_id', None)

    # Form for adding new data
    with st.form("new_data_form"):
        # Suggestion selection dropdown
        if suggestions:
            st.subheader("Suggestion Reference (Optional)")
            suggestion_options = ["None - Manual Entry"] + [suggestions[sid]['label'] for sid in suggestions.keys()]
            suggestion_ids = [None] + list(suggestions.keys())
            
            # Find default index
            default_index = 0
            if default_suggestion and default_suggestion in suggestions:
                default_index = suggestion_ids.index(default_suggestion)
            
            selected_suggestion_option = st.selectbox(
                "Select a suggestion to prefill parameters (you can still modify them):",
                options=suggestion_options,
                index=default_index,
                key="suggestion_selectbox"
            )
            
            # Get the selected suggestion ID
            selected_suggestion_id = None if selected_suggestion_option == "None - Manual Entry" else suggestion_ids[suggestion_options.index(selected_suggestion_option)]
            
            # Update defaults if a suggestion is selected
            if selected_suggestion_id and selected_suggestion_id in suggestions:
                suggestion_data = suggestions[selected_suggestion_id]
                default_temp = suggestion_data['temp']
                default_humidity = suggestion_data['humidity']
                default_layer_count = suggestion_data['layer_count']
                default_layer_height = suggestion_data['slicer_layer_height']
                default_layer_width = suggestion_data['slicer_layer_width']
                default_nozzle_speed = suggestion_data['slicer_nozzle_speed']
                default_extrusion_multiplier = suggestion_data['slicer_extrusion_multiplier']
        else:
            selected_suggestion_id = None
        
        # Create two columns for the form layout
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            st.subheader("Measurements")
            height_1 = st.number_input("Height 1 (mm)", min_value=0.0, step=0.01, format="%.2f")
            height_2 = st.number_input("Height 2 (mm)", min_value=0.0, step=0.01, format="%.2f")
            height_3 = st.number_input("Height 3 (mm)", min_value=0.0, step=0.01, format="%.2f")
            width_1 = st.number_input("Width 1 (mm)", min_value=0.0, step=0.01, format="%.2f")
            width_2 = st.number_input("Width 2 (mm)", min_value=0.0, step=0.01, format="%.2f")
            width_3 = st.number_input("Width 3 (mm)", min_value=0.0, step=0.01, format="%.2f")

        with form_col2:
            st.subheader("Parameters")
            new_temp = st.number_input("Temperature (°C)", min_value=0.0, value=default_temp, step=0.1, format="%.1f",
                                       key="form_temp")
            new_humidity = st.number_input("Humidity (%)", min_value=1.0, max_value=100.0, value=default_humidity, step=1.0,
                                           format="%.1f", key="form_humidity")
            new_layer_count = st.number_input("Layer Count", min_value=1, max_value=10, value=default_layer_count,
                                             step=1, format="%d", key="form_layer_count")
            new_layer_height = st.number_input("Slicer Layer Height", min_value=0.0, value=default_layer_height,
                                               step=0.1, format="%.1f", key="form_layer_height")
            new_layer_width = st.number_input("Slicer Layer Width", min_value=0.0, value=default_layer_width, step=0.1,
                                              format="%.1f", key="form_layer_width")
            new_nozzle_speed = st.number_input("Slicer Nozzle Speed", min_value=0.0, value=default_nozzle_speed,
                                               step=1.0, format="%.1f", key="form_nozzle_speed")
            new_extrusion_multiplier = st.number_input("Extrusion Multiplier", min_value=0.0,
                                                       value=default_extrusion_multiplier, step=0.01, format="%.2f",
                                                       key="form_extrusion_multiplier")

        # Submit button
        submitted = st.form_submit_button("Add Data Point")

        if submitted:
            new_data_point = {
                'height_1': height_1,
                'height_2': height_2,
                'height_3': height_3,
                'width_1': width_1,
                'width_2': width_2,
                'width_3': width_3,
                'temp': new_temp,
                'humidity': new_humidity,
                'layer_count': new_layer_count,
                'slicer_layer_height': new_layer_height,
                'slicer_layer_width': new_layer_width,
                'slicer_nozzle_speed': new_nozzle_speed,
                'slicer_extrusion_multiplier': new_extrusion_multiplier,
                'suggestion_id': selected_suggestion_id
            }

            # Validate the data point
            errors = validate_new_data_point(new_data_point)

            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Add data to database
                add_data_point(new_data_point)

                # Clear any transferred parameters only after successful submission
                stored_params_to_clear = [
                    'stored_temp', 'stored_humidity', 'stored_layer_count',
                    'stored_layer_height', 'stored_layer_width', 'stored_nozzle_speed', 
                    'stored_extrusion_multiplier', 'selected_suggestion_id', 
                    'prefilled_from_suggestion', 'prefilled_from_prediction'
                ]
                
                for param in stored_params_to_clear:
                    if param in st.session_state:
                        del st.session_state[param]

                # Reset prediction in session state
                st.session_state.prediction = None

                # Show success message
                st.success("Data point added successfully!")

                # Refresh data display
                st.rerun()
        
elif page == "Optimization History":
    st.header("Bayesian Optimization History")
    
    # Get suggested experiments from the database
    suggested_df = get_suggested_experiments()
    
    if not suggested_df.empty:
        # Display summary statistics
        st.subheader("Summary Statistics")
        
        # Group by dataset size to see how suggestions improved over time
        grouped = suggested_df.groupby('dataset_size').agg({
            'total_mismatch': ['mean', 'min', 'std'],
            'id': 'count'
        }).reset_index()
        grouped.columns = ['dataset_size', 'avg_mismatch', 'min_mismatch', 'std_mismatch', 'suggestion_count']
        
        # Display the summary table
        st.dataframe(grouped)
        
        # Visualization of how total_mismatch improves with dataset size
        st.subheader("Optimization Progress")
        
        import plotly.express as px
        
        # Create scatter plot of total mismatch vs dataset size
        fig1 = px.scatter(
            suggested_df, 
            x='dataset_size', 
            y='total_mismatch', 
            color='suggestion_type',
            title='Dimension Mismatch vs Dataset Size',
            labels={
                'total_mismatch': 'Total Dimension Mismatch (mm)',
                'dataset_size': 'Dataset Size',
                'suggestion_type': 'Suggestion Type'
            },
            opacity=0.7
        )
        
        fig1.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig1, use_container_width=True)
        
        # Create box plot to compare the two suggestion types
        fig2 = px.box(
            suggested_df, 
            x='suggestion_type', 
            y='total_mismatch',
            title='Mismatch Distribution by Suggestion Type',
            labels={
                'total_mismatch': 'Total Dimension Mismatch (mm)',
                'suggestion_type': 'Suggestion Type'
            }
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display raw data with filters
        st.subheader("Suggested Experiments Data")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            dataset_sizes = sorted(suggested_df['dataset_size'].unique())
            selected_size = st.selectbox(
                "Filter by Dataset Size", 
                options=["All"] + dataset_sizes,
                index=0
            )
        
        with col2:
            suggestion_types = sorted(suggested_df['suggestion_type'].unique())
            selected_type = st.selectbox(
                "Filter by Suggestion Type", 
                options=["All"] + suggestion_types,
                index=0
            )
        
        # Apply filters
        filtered_df = suggested_df.copy()
        if selected_size != "All":
            filtered_df = filtered_df[filtered_df['dataset_size'] == selected_size]
        
        if selected_type != "All":
            filtered_df = filtered_df[filtered_df['suggestion_type'] == selected_type]
        
        # Display the filtered data with integrated action buttons
        if not filtered_df.empty:
            st.subheader("Suggested Experiments Data")
            st.markdown("Each suggestion is displayed below with a 'Use' button to transfer parameters to 'Add New Data'.")
            
            # Create a container for each suggestion row
            for idx, (_, row) in enumerate(filtered_df.iterrows()):
                with st.container():
                    # Create columns: data display (wider) + action button (narrower)
                    data_col, action_col = st.columns([5, 1])
                    
                    with data_col:
                        # Display row data in a structured format
                        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                        
                        with info_col1:
                            st.write(f"**Suggestion #{row['id']}**")
                            st.write(f"📊 Dataset: {row['dataset_size']} records")
                            st.write(f"🔧 Type: {row['suggestion_type']}")
                            if pd.notna(row.get('timestamp')):
                                st.write(f"⏰ {row['timestamp'][:16]}")
                        
                        with info_col2:
                            st.write("**🌡️ Environment**")
                            st.write(f"Temp: {row['temp']:.1f}°C")
                            st.write(f"Humidity: {row['humidity']:.1f}%")
                        
                        with info_col3:
                            st.write("**⚙️ Slicer Settings**")
                            if 'layer_count' in row and pd.notna(row['layer_count']):
                                st.write(f"Layers: {int(row['layer_count'])}")
                            st.write(f"Height: {row['slicer_layer_height']:.1f}mm")
                            st.write(f"Width: {row['slicer_layer_width']:.1f}mm")
                            st.write(f"Speed: {row['slicer_nozzle_speed']:.1f}")
                            st.write(f"Multiplier: {row['slicer_extrusion_multiplier']:.2f}")
                        
                        with info_col4:
                            st.write("**📏 Predictions**")
                            st.write(f"Width: {row['predicted_width']:.2f}mm")
                            st.write(f"Height: {row['predicted_height']:.2f}mm")
                            st.write(f"📉 Mismatch: {row['total_mismatch']:.3f}mm")
                    
                    with action_col:
                        # Vertical centering with empty space
                        st.write("")
                        st.write("")
                        
                        # Action button for this row
                        button_key = f"use_suggestion_inline_{row['id']}"
                        if st.button(
                            "Use it", 
                            key=button_key, 
                            help="Transfer these parameters to 'Add New Data'",
                            type="primary",
                            use_container_width=True
                        ):
                            # Store parameters and navigate to Add New Data
                            store_history_suggestion_params_for_form(row.to_dict(), row['id'])
                            st.rerun()
                
                # Add a subtle separator between rows
                if idx < len(filtered_df) - 1:
                    st.divider()
        else:
            st.info("No suggestions match the selected filters.")
        
        # Allow export to CSV
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="optimization_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No optimization history available yet. Generate some experiment suggestions to see data here.")
