import streamlit as st


def add_custom_css():
    """Add custom CSS to streamlit app"""
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
        }
        /* Styling for metrics and cards */
        .metric-card {
            background-color: white;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


def format_params_for_display(params):
    """Format parameters dictionary for display"""
    formatted = []

    for key, value in params.items():
        # Format the key by replacing underscores with spaces and capitalizing
        formatted_key = key.replace('_', ' ').title()

        # Format the value based on parameter type
        if 'temp' in key:
            formatted_value = f"{value:.1f} °C"
        elif 'humidity' in key:
            formatted_value = f"{value:.2f}"
        elif 'height' in key or 'width' in key or 'multiplier' in key:
            formatted_value = f"{value:.2f} mm"
        elif 'speed' in key:
            formatted_value = f"{value:.1f} mm/s"
        else:
            formatted_value = f"{value}"

        formatted.append((formatted_key, formatted_value))

    return formatted


def validate_new_data_point(data_point):
    """Validate the new data point before adding to database"""
    errors = []

    # Check for non-zero measurements
    for key in ['height_1', 'height_2', 'height_3', 'width_1', 'width_2', 'width_3']:
        if data_point[key] <= 0:
            errors.append(f"{key.replace('_', ' ').title()} must be greater than zero")

    # Check for valid parameter ranges
    if data_point['humidity'] < 0 or data_point['humidity'] > 100:
        errors.append("Humidity must be between 0 and 100")

    if data_point['temp'] <= 0:
        errors.append("Temperature must be greater than zero")

    if data_point['slicer_extrusion_multiplier'] <= 0:
        errors.append("Extrusion multiplier must be greater than zero")

    return errors