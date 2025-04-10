import streamlit as st
import pandas as pd

# Import modules
from database import init_db, get_data, get_data_for_display, add_data_point, delete_data_point, \
    delete_multiple_data_points
from models import train_gpr_models, make_prediction
from visualization import create_scatter_plot, create_summary_stats
from utils import add_custom_css, validate_new_data_point

# Set page config
st.set_page_config(
    page_title="Robocasting Experiments",
    page_icon="🤖",
    layout="wide"
)

# Add custom CSS
add_custom_css()

# Initialize the database
init_db()

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'sidebar_page' not in st.session_state:
    st.session_state.sidebar_page = "Data Explorer"


# Function to store prediction parameters for add_data form
def store_params_for_form():
    st.session_state.stored_temp = st.session_state.temp_input
    st.session_state.stored_humidity = st.session_state.humidity_input
    st.session_state.stored_layer_height = st.session_state.layer_height_input
    st.session_state.stored_layer_width = st.session_state.layer_width_input
    st.session_state.stored_nozzle_speed = st.session_state.nozzle_speed_input
    st.session_state.stored_extrusion_multiplier = st.session_state.extrusion_multiplier_input
    st.session_state.show_add_data = True
    # Update sidebar selection
    st.session_state.sidebar_page = "Add New Data"


# App title
st.title("🤖 Robocasting Experiments")

# Add sidebar navigation - use the saved state for the default value
page = st.sidebar.radio("Navigation", ["Data Explorer", "Predictions", "Add New Data"],
                        index=["Data Explorer", "Predictions", "Add New Data"].index(st.session_state.sidebar_page),
                        key="navigation")

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
    # st.header("Dataset")

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
    else:
        st.info("No data available. Add data points to see summary and visualizations.")

elif page == "Predictions":
    st.header("Gaussian Process Regression Predictions")

    # Load data - use the full data with IDs for model training
    data = get_data()

    if not data.empty:
        # Train models
        width_model, height_model = train_gpr_models(data)

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

            humidity = st.number_input("Humidity",
                                       min_value=float(data['humidity'].min() - 0.05),
                                       max_value=float(data['humidity'].max() + 0.05),
                                       value=float(data['humidity'].mean()),
                                       step=0.01,
                                       format="%.2f",
                                       key="humidity_input")

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
                # st.progress((prediction['width'] / 5.0))  # Show a progress bar with relative width
                st.markdown(f"**Range:** {width_range[0]:.2f} mm to {width_range[1]:.2f} mm")

            with result_col2:
                st.metric("Predicted Avg Height", f"{prediction['height']:.2f} mm")
                st.metric("Height Uncertainty", f"±{prediction['height_uncertainty']:.3f} mm")

                # Create a simple visualization with confidence interval
                height_range = (prediction['height'] - prediction['height_uncertainty'],
                                prediction['height'] + prediction['height_uncertainty'])
                st.markdown("**95% Confidence Interval:**")
                # st.progress((prediction['height'] / 2.0))  # Show a progress bar with relative height
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
                st.progress((prediction['width'] / 5.0))  # Show a progress bar with relative width
                st.markdown(f"**Range:** {width_range[0]:.2f} mm to {width_range[1]:.2f} mm")

            with result_col2:
                st.metric("Predicted Avg Height", f"{prediction['height']:.2f} mm")
                st.metric("Height Uncertainty", f"±{prediction['height_uncertainty']:.3f} mm")

                # Create a simple visualization with confidence interval
                height_range = (prediction['height'] - prediction['height_uncertainty'],
                                prediction['height'] + prediction['height_uncertainty'])
                st.markdown("**95% Confidence Interval:**")
                st.progress((prediction['height'] / 2.0))  # Show a progress bar with relative height
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

elif page == "Add New Data":
    st.header("Add New Data Point")

    # Use parameters from Prediction tab if available
    default_temp = st.session_state.get('stored_temp', 20.0)
    default_humidity = st.session_state.get('stored_humidity', 0.4)
    default_layer_height = st.session_state.get('stored_layer_height', 0.8)
    default_layer_width = st.session_state.get('stored_layer_width', 1.5)
    default_nozzle_speed = st.session_state.get('stored_nozzle_speed', 8.0)
    default_extrusion_multiplier = st.session_state.get('stored_extrusion_multiplier', 1.0)

    # Form for adding new data
    with st.form("new_data_form"):
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
            new_humidity = st.number_input("Humidity", min_value=0.0, max_value=1.0, value=default_humidity, step=0.01,
                                           format="%.2f", key="form_humidity")
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
                'slicer_layer_height': new_layer_height,
                'slicer_layer_width': new_layer_width,
                'slicer_nozzle_speed': new_nozzle_speed,
                'slicer_extrusion_multiplier': new_extrusion_multiplier
            }

            # Validate the data point
            errors = validate_new_data_point(new_data_point)

            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Add data to database
                add_data_point(new_data_point)

                # Clear any transferred parameters
                if 'stored_temp' in st.session_state:
                    del st.session_state.stored_temp
                if 'stored_humidity' in st.session_state:
                    del st.session_state.stored_humidity
                if 'stored_layer_height' in st.session_state:
                    del st.session_state.stored_layer_height
                if 'stored_layer_width' in st.session_state:
                    del st.session_state.stored_layer_width
                if 'stored_nozzle_speed' in st.session_state:
                    del st.session_state.stored_nozzle_speed
                if 'stored_extrusion_multiplier' in st.session_state:
                    del st.session_state.stored_extrusion_multiplier

                # Reset prediction in session state
                st.session_state.prediction = None

                # Show success message
                st.success("Data point added successfully!")

                # Refresh data display
                st.rerun()