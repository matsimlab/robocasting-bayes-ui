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

    # Create table
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
        slicer_extrusion_multiplier REAL
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
        slicer_nozzle_speed, slicer_extrusion_multiplier
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data_point['height_1'], data_point['height_2'], data_point['height_3'],
        data_point['width_1'], data_point['width_2'], data_point['width_3'],
        data_point['temp'], data_point['humidity'],
        data_point['slicer_layer_height'], data_point['slicer_layer_width'],
        data_point['slicer_nozzle_speed'], data_point['slicer_extrusion_multiplier']
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