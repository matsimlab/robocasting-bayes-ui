import plotly.express as px


def create_scatter_plot(data, x_col, y_col, title):
    """Create a scatter plot for visualizing relationships between parameters"""
    fig = px.scatter(
        data, x=x_col, y=y_col,
        title=title,
        labels={
            x_col: x_col.replace('_', ' ').title(),
            y_col: y_col.replace('_', ' ').title()
        },
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

    # Adjust layout to prevent text overlap
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),  # Increased bottom and left margins
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        title_x=0.5,  # Center the title
        font=dict(size=12),  # Slightly larger font
        xaxis=dict(
            title_standoff=20,  # More space for x-axis title
            tickangle=-45 if len(x_col) > 10 else 0  # Angle tick labels if the column name is long
        ),
        yaxis=dict(
            title_standoff=20  # More space for y-axis title
        )
    )

    return fig


def create_summary_stats(data):
    """Return a dictionary containing summary statistics"""
    if data.empty:
        return None

    # Calculate average height and width for each row
    data['avg_height'] = (data['height_1'] + data['height_2'] + data['height_3']) / 3
    data['avg_width'] = (data['width_1'] + data['width_2'] + data['width_3']) / 3

    return {
        'count': len(data),
        'avg_width': data['avg_width'].mean(),
        'avg_height': data['avg_height'].mean(),
        'avg_temp': data['temp'].mean(),
        'min_width': data['avg_width'].min(),
        'max_width': data['avg_width'].max(),
        'min_height': data['avg_height'].min(),
        'max_height': data['avg_height'].max(),
        'min_temp': data['temp'].min(),
        'max_temp': data['temp'].max()
    }