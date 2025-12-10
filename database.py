"""SQLite Database Operations for Robocasting Experiments

Manages two tables:
1. experiments: Dimensional measurements (width/height) and process parameters
   - Supports soft-delete via 'archived' flag to preserve data while excluding from training
   - Tracks suggestion_id to link experiments back to Bayesian optimization suggestions
   
2. suggested_experiments: Bayesian optimization history
   - Records all suggested parameter combinations
   - Stores predictions, uncertainties, and dimension mismatches
   - Tracks dataset size at time of suggestion for performance analysis

Database Schema:
    experiments:
        id (PK), height_1/2/3, width_1/2/3, temp, humidity, layer_count,
        slicer_layer_height, slicer_layer_width, slicer_nozzle_speed,
        slicer_extrusion_multiplier, suggestion_id (FK), archived (0/1)
    
    suggested_experiments:
        id (PK), dataset_size, suggestion_type, temp, humidity, layer_count,
        slicer_*, predicted_width/height, uncertainties, mismatches,
        target_height/width, timestamp

Auto-initialization:
    On first run, if cleaned_df.csv exists, data is automatically imported.
    Database persists in data/robocasting.db (or /app/data/robocasting.db in Docker).

Author: Nazarii Mediukh
Institution: IPMS NASU
"""

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
        layer_count INTEGER,
        slicer_layer_height REAL,
        slicer_layer_width REAL,
        slicer_nozzle_speed REAL,
        slicer_extrusion_multiplier REAL,
        suggestion_id INTEGER,
        archived INTEGER DEFAULT 0,
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
    
    # Add archived column if it doesn't exist (for existing databases)
    try:
        c.execute('ALTER TABLE experiments ADD COLUMN archived INTEGER DEFAULT 0')
        conn.commit()
        print("Added archived column to experiments table")
        
        # Set default archived = 0 for existing records that don't have it
        c.execute('UPDATE experiments SET archived = 0 WHERE archived IS NULL')
        conn.commit()
        print("Set default archived = 0 for existing records")
        
    except sqlite3.OperationalError:
        # Column already exists or other error - still try to update NULL values
        try:
            c.execute('UPDATE experiments SET archived = 0 WHERE archived IS NULL')
            if c.rowcount > 0:
                conn.commit()
                print(f"Updated {c.rowcount} records with default archived = 0")
        except:
            pass
    
    # Add layer_count column if it doesn't exist (for existing databases)
    try:
        c.execute('ALTER TABLE experiments ADD COLUMN layer_count INTEGER')
        conn.commit()
        print("Added layer_count column to experiments table")
        
        # Set default layer_count = 1 for existing records that don't have it
        c.execute('UPDATE experiments SET layer_count = 1 WHERE layer_count IS NULL')
        conn.commit()
        print("Set default layer_count = 1 for existing records")
        
    except sqlite3.OperationalError:
        # Column already exists or other error - still try to update NULL values
        try:
            c.execute('UPDATE experiments SET layer_count = 1 WHERE layer_count IS NULL')
            if c.rowcount > 0:
                conn.commit()
                print(f"Updated {c.rowcount} records with default layer_count = 1")
        except:
            pass
    
    # Add layer_count column to suggested_experiments if it doesn't exist
    try:
        c.execute('ALTER TABLE suggested_experiments ADD COLUMN layer_count INTEGER')
        conn.commit()
        print("Added layer_count column to suggested_experiments table")
    except sqlite3.OperationalError:
        # Column already exists or other error - that's fine
        pass
    
    # Add target columns to suggested_experiments if they don't exist
    try:
        c.execute('ALTER TABLE suggested_experiments ADD COLUMN target_height REAL')
        conn.commit()
        print("Added target_height column to suggested_experiments table")
    except sqlite3.OperationalError:
        pass
    
    try:
        c.execute('ALTER TABLE suggested_experiments ADD COLUMN target_width REAL')
        conn.commit()
        print("Added target_width column to suggested_experiments table")
    except sqlite3.OperationalError:
        pass
    
    # Create suggested_experiments table to track Bayesian optimization suggestions
    c.execute('''
    CREATE TABLE IF NOT EXISTS suggested_experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_size INTEGER,
        suggestion_type TEXT,
        temp REAL,
        humidity REAL,
        layer_count INTEGER,
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

    # Check if table is empty or if we need to reimport due to layer_count
    c.execute("SELECT COUNT(*) FROM experiments")
    count = c.fetchone()[0]
    
    # Check if we have layer_count data
    has_layer_count_data = False
    if count > 0:
        try:
            c.execute("SELECT COUNT(*) FROM experiments WHERE layer_count IS NOT NULL")
            layer_count_records = c.fetchone()[0]
            has_layer_count_data = layer_count_records > 0
        except:
            has_layer_count_data = False
    
    print(f"Database status: {count} total records, layer_count data: {has_layer_count_data}")

    # If table is empty or missing layer_count data, import data from CSV
    if (count == 0 or not has_layer_count_data) and os.path.exists('cleaned_df.csv'):
        print("Importing/updating data from CSV...")
        
        # If we're updating existing data, clear the table first
        if count > 0 and not has_layer_count_data:
            print("Clearing existing data to reimport with layer_count")
            c.execute("DELETE FROM experiments")
            conn.commit()
        
        df = pd.read_csv('cleaned_df.csv')
        print(f"CSV contains {len(df)} records with columns: {list(df.columns)}")

        # Verify CSV has layer_count column
        if 'layer_count' not in df.columns:
            print("WARNING: CSV does not contain layer_count column!")
            df['layer_count'] = 1  # Add default value
        
        # Insert data into SQLite
        for _, row in df.iterrows():
            c.execute('''
            INSERT INTO experiments (
                height_1, height_2, height_3,
                width_1, width_2, width_3,
                temp, humidity, layer_count,
                slicer_layer_height, slicer_layer_width,
                slicer_nozzle_speed, slicer_extrusion_multiplier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['height_1'], row['height_2'], row['height_3'],
                row['width_1'], row['width_2'], row['width_3'],
                row['temp'], row['humidity'], row['layer_count'],
                row['slicer_layer_height'], row['slicer_layer_width'],
                row['slicer_nozzle_speed'], row['slicer_extrusion_multiplier']
            ))

        print(f"Successfully imported {len(df)} records from CSV to database.")

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


def get_data(include_archived=False):
    """Get data from the database
    
    Args:
        include_archived: If False (default), only returns non-archived records.
                         If True, returns all records including archived ones.
    """
    conn = sqlite3.connect(get_db_path())
    if include_archived:
        query = "SELECT * FROM experiments"
    else:
        query = "SELECT * FROM experiments WHERE archived = 0"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_data_for_display(include_archived=False):
    """Get data from the database with the internal ID hidden
    
    Args:
        include_archived: If False (default), only returns non-archived records.
                         If True, returns all records including archived ones.
    """
    conn = sqlite3.connect(get_db_path())
    if include_archived:
        query = "SELECT * FROM experiments"
    else:
        query = "SELECT * FROM experiments WHERE archived = 0"
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
        temp, humidity, layer_count,
        slicer_layer_height, slicer_layer_width,
        slicer_nozzle_speed, slicer_extrusion_multiplier,
        suggestion_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data_point['height_1'], data_point['height_2'], data_point['height_3'],
        data_point['width_1'], data_point['width_2'], data_point['width_3'],
        data_point['temp'], data_point['humidity'], data_point['layer_count'],
        data_point['slicer_layer_height'], data_point['slicer_layer_width'],
        data_point['slicer_nozzle_speed'], data_point['slicer_extrusion_multiplier'],
        data_point.get('suggestion_id', None)
    ))

    conn.commit()
    conn.close()


def archive_data_point(id):
    """Archive a data point by setting archived = 1
    
    Args:
        id: ID of the experiment to archive
    
    Returns:
        True if a row was archived, False otherwise
    """
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    c.execute("UPDATE experiments SET archived = 1 WHERE id = ? AND archived = 0", (id,))

    # Get the number of rows affected by the update operation
    rows_archived = c.rowcount

    conn.commit()
    conn.close()

    return rows_archived > 0


def unarchive_data_point(id):
    """Unarchive a data point by setting archived = 0
    
    Args:
        id: ID of the experiment to unarchive
    
    Returns:
        True if a row was unarchived, False otherwise
    """
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    c.execute("UPDATE experiments SET archived = 0 WHERE id = ? AND archived = 1", (id,))

    # Get the number of rows affected by the update operation
    rows_unarchived = c.rowcount

    conn.commit()
    conn.close()

    return rows_unarchived > 0


def archive_multiple_data_points(ids):
    """Archive multiple data points by setting archived = 1
    
    Args:
        ids: List of experiment IDs to archive
    
    Returns:
        Number of rows archived
    """
    if not ids:
        return 0

    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    # Create placeholders for the SQL query
    placeholders = ','.join(['?'] * len(ids))
    c.execute(f"UPDATE experiments SET archived = 1 WHERE id IN ({placeholders}) AND archived = 0", ids)

    # Get the number of rows affected by the update operation
    rows_archived = c.rowcount

    conn.commit()
    conn.close()

    return rows_archived


def delete_data_point(id):
    """Delete a data point from the database by ID (HARD DELETE - use archive_data_point instead)
    
    Note: This function performs a permanent deletion. Consider using archive_data_point() instead.
    """
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()

    c.execute("DELETE FROM experiments WHERE id = ?", (id,))

    # Get the number of rows affected by the delete operation
    rows_deleted = c.rowcount

    conn.commit()
    conn.close()

    return rows_deleted > 0


def delete_multiple_data_points(ids):
    """Delete multiple data points from the database by IDs (HARD DELETE - use archive_multiple_data_points instead)
    
    Note: This function performs a permanent deletion. Consider using archive_multiple_data_points() instead.
    """
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
        layer_count,
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
        total_mismatch,
        target_height,
        target_width
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        dataset_size,
        suggestion_type,
        suggestion_data['temp'],
        suggestion_data['humidity'],
        suggestion_data['layer_count'],
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
        suggestion_data['total_mismatch'],
        suggestion_data.get('target_height', None),
        suggestion_data.get('target_width', None)
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
    SELECT id, temp, humidity, layer_count, slicer_layer_height, slicer_layer_width, 
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
        label = (f"Suggestion #{row['id']} - T:{row['temp']:.1f}°C, LC:{row['layer_count']}, "
                f"H:{row['slicer_layer_height']:.1f}mm, W:{row['slicer_layer_width']:.1f}mm "
                f"(Pred: W={row['predicted_width']:.2f}, H={row['predicted_height']:.2f})")
        suggestions[row['id']] = {
            'label': label,
            'temp': row['temp'],
            'humidity': row['humidity'],
            'layer_count': row['layer_count'],
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