import sqlite3
import pandas as pd
import os
import streamlit as st


def init_db():
    """Initialize the SQLite database and create table if it doesn't exist"""
    # Create data directory if it doesn't exist
    # Check if we're running in Docker or locally
    import os
    
    # Determine the database path - use local path for development
    if os.path.exists('/app'):
        # Docker environment
        db_dir = '/app/data'
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, 'robocasting.db')
    else:
        # Local development environment
        db_dir = 'data'
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, 'robocasting.db')
        
    # Log the database path for debugging
    print(f"Using database at: {db_path}")
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create experiments table
    c.execute('''
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        height_1 REAL,
        height_2 REAL,
        height_3 REAL,
        width_1 REAL,
        width_2 REAL,
        width_3 REAL,
        temp REAL,
        humidity REAL,
        slicer_layer_height REAL,
        slicer_layer_width REAL,
        slicer_nozzle_speed REAL,
        slicer_extrusion_multiplier REAL,
        suggestion_id INTEGER,
        FOREIGN KEY (suggestion_id) REFERENCES suggested_experiments (id)
    )
    ''')
    
    # Add suggestion_id column if it doesn't exist (for existing databases)
    try:
        c.execute('ALTER TABLE experiments ADD COLUMN suggestion_id INTEGER')
        conn.commit()
        print("Added suggestion_id column to experiments table")
    except sqlite3.OperationalError:
        # Column already exists or other error - that's fine
        pass
    
    # Create suggested_experiments table to track Bayesian optimization suggestions
    c.execute('''
    CREATE TABLE IF NOT EXISTS suggested_experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_size INTEGER,
        suggestion_type TEXT,
        temp REAL,
        humidity REAL,
        slicer_layer_height REAL,
        slicer_layer_width REAL,
        slicer_nozzle_speed REAL,
        slicer_extrusion_multiplier REAL,
        predicted_width REAL,
        predicted_height REAL,
        width_uncertainty REAL,
        height_uncertainty REAL,
        width_mismatch REAL,
        height_mismatch REAL,
        total_mismatch REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Check if table is empty
    c.execute("SELECT COUNT(*) FROM experiments")
    count = c.fetchone()[0]

    # If table is empty, import data from CSV
    if count == 0 and os.path.exists('cleaned_df.csv'):
        df = pd.read_csv('cleaned_df.csv')

        # Insert data into SQLite
        for _, row in df.iterrows():
            c.execute('''
            INSERT INTO experiments (
                height_1, height_2, height_3,
                width_1, width_2, width_3,
                temp, humidity,
                slicer_layer_height, slicer_layer_width,
                slicer_nozzle_speed, slicer_extrusion_multiplier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['height_1'], row['height_2'], row['height_3'],
                row['width_1'], row['width_2'], row['width_3'],
                row['temp'], row['humidity'],
                row['slicer_layer_height'], row['slicer_layer_width'],
                row['slicer_nozzle_speed'], row['slicer_extrusion_multiplier']
            ))

        st.success(f"Imported {len(df)} records from CSV to database.")

    conn.commit()
    conn.close()


def get_db_path():
    """Helper function to determine the database path"""
    import os
    
    if os.path.exists('/app'):
        # Docker environment
        db_dir = '/app/data'
        os.makedirs(db_dir, exist_ok=True)
        return os.path.join(db_dir, 'robocasting.db')
    else:
        # Local development environment
        db_dir = 'data'
        os.makedirs(db_dir, exist_ok=True)
        return os.path.join(db_dir, 'robocasting.db')


def get_data():
    """Get all data from the database"""
    conn = sqlite3.connect(get_db_path())
    query = "SELECT * FROM experiments"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_data_for_display():
    """Get all data from the database with the internal ID hidden"""
    conn = sqlite3.connect(get_db_path())
    query = "SELECT * FROM experiments"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Store the ID separately for reference but hide from display
    display_df = df.copy()
    if 'id' in display_df.columns:
        display_df = display_df.drop('id', axis=1)

    return df, display_df


def add_data_point(data_point):
    """Add a new data point to the database"""
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    c.execute('''
    INSERT INTO experiments (
        height_1, height_2, height_3,
        width_1, width_2, width_3,
        temp, humidity,
        slicer_layer_height, slicer_layer_width,
        slicer_nozzle_speed, slicer_extrusion_multiplier,
        suggestion_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data_point['height_1'], data_point['height_2'], data_point['height_3'],
        data_point['width_1'], data_point['width_2'], data_point['width_3'],
        data_point['temp'], data_point['humidity'],
        data_point['slicer_layer_height'], data_point['slicer_layer_width'],
        data_point['slicer_nozzle_speed'], data_point['slicer_extrusion_multiplier'],
        data_point.get('suggestion_id', None)
    ))

    conn.commit()
    conn.close()


def delete_data_point(id):
    """Delete a data point from the database by ID"""
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    c.execute("DELETE FROM experiments WHERE id = ?", (id,))

    # Get the number of rows affected by the delete operation
    rows_deleted = c.rowcount

    conn.commit()
    conn.close()

    return rows_deleted > 0


def delete_multiple_data_points(ids):
    """Delete multiple data points from the database by IDs"""
    if not ids:
        return 0

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    # Create placeholders for the SQL query
    placeholders = ','.join(['?'] * len(ids))
    c.execute(f"DELETE FROM experiments WHERE id IN ({placeholders})", ids)

    # Get the number of rows affected by the delete operation
    rows_deleted = c.rowcount

    conn.commit()
    conn.close()

    return rows_deleted


def add_suggested_experiment(suggestion_data, dataset_size, suggestion_type="single_point"):
    """Add a suggested experiment from Bayesian optimization to the database
    
    Args:
        suggestion_data: Dictionary containing the suggested parameters and predictions
        dataset_size: Size of the dataset when the suggestion was made
        suggestion_type: Type of suggestion ('single_point' or 'design_space_exploration')
    
    Returns:
        ID of the newly inserted record
    """
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    c.execute('''
    INSERT INTO suggested_experiments (
        dataset_size,
        suggestion_type,
        temp,
        humidity,
        slicer_layer_height,
        slicer_layer_width,
        slicer_nozzle_speed,
        slicer_extrusion_multiplier,
        predicted_width,
        predicted_height,
        width_uncertainty,
        height_uncertainty,
        width_mismatch,
        height_mismatch,
        total_mismatch
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        dataset_size,
        suggestion_type,
        suggestion_data['temp'],
        suggestion_data['humidity'],
        suggestion_data['slicer_layer_height'],
        suggestion_data['slicer_layer_width'],
        suggestion_data['slicer_nozzle_speed'],
        suggestion_data['slicer_extrusion_multiplier'],
        suggestion_data['predicted_width'],
        suggestion_data['predicted_height'],
        suggestion_data.get('width_uncertainty', 0.0),
        suggestion_data.get('height_uncertainty', 0.0),
        suggestion_data['width_mismatch'],
        suggestion_data['height_mismatch'],
        suggestion_data['total_mismatch']
    ))
    
    # Get the ID of the newly inserted row
    last_id = c.lastrowid
    
    conn.commit()
    conn.close()
    
    return last_id


def get_suggested_experiments(limit=None):
    """Get suggested experiments from the database with optional limit
    
    Args:
        limit: Optional maximum number of records to return
    
    Returns:
        DataFrame containing suggested experiments
    """
    conn = sqlite3.connect(get_db_path())
    
    if limit:
        query = f"SELECT * FROM suggested_experiments ORDER BY timestamp DESC LIMIT {limit}"
    else:
        query = "SELECT * FROM suggested_experiments ORDER BY timestamp DESC"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def get_suggested_experiments_for_dropdown():
    """Get suggested experiments formatted for dropdown selection
    
    Returns:
        Dictionary mapping suggestion IDs to descriptive labels
    """
    conn = sqlite3.connect(get_db_path())
    
    query = '''
    SELECT id, temp, humidity, slicer_layer_height, slicer_layer_width, 
           slicer_nozzle_speed, slicer_extrusion_multiplier, 
           predicted_width, predicted_height, total_mismatch, timestamp
    FROM suggested_experiments 
    ORDER BY timestamp DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    suggestions = {}
    for _, row in df.iterrows():
        # Create a descriptive label for the dropdown
        label = (f"Suggestion #{row['id']} - T:{row['temp']:.1f}°C, "
                f"H:{row['slicer_layer_height']:.1f}mm, W:{row['slicer_layer_width']:.1f}mm "
                f"(Pred: W={row['predicted_width']:.2f}, H={row['predicted_height']:.2f})")
        suggestions[row['id']] = {
            'label': label,
            'temp': row['temp'],
            'humidity': row['humidity'],
            'slicer_layer_height': row['slicer_layer_height'],
            'slicer_layer_width': row['slicer_layer_width'],
            'slicer_nozzle_speed': row['slicer_nozzle_speed'],
            'slicer_extrusion_multiplier': row['slicer_extrusion_multiplier'],
            'predicted_width': row['predicted_width'],
            'predicted_height': row['predicted_height']
        }
    
    return suggestions


def get_suggestion_by_id(suggestion_id):
    """Get a specific suggestion by ID
    
    Args:
        suggestion_id: ID of the suggestion to retrieve
    
    Returns:
        Dictionary containing suggestion data or None if not found
    """
    conn = sqlite3.connect(get_db_path())
    
    query = '''
    SELECT * FROM suggested_experiments WHERE id = ?
    '''
    
    cursor = conn.cursor()
    cursor.execute(query, (suggestion_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row))
    else:
        return None