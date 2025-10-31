#!/usr/bin/env python3
"""
Convert dataset/ds.csv to match cleaned_df.csv format
- Remove: number, date, nozzle_diameter columns
- Keep: layer_count (as specifically requested)
- Convert European decimal notation (commas) to standard notation (dots)
"""

import pandas as pd
import os

def convert_european_decimal(value):
    """Convert European decimal notation (commas) to standard notation (dots)"""
    if isinstance(value, str):
        # Remove quotes and convert comma to dot
        return value.strip('"').replace(',', '.')
    return value

def convert_dataset():
    """Main conversion function"""
    
    # File paths
    input_file = 'dataset/ds.csv'
    output_file = 'dataset/ds_converted.csv'
    
    print(f"Reading dataset from: {input_file}")
    
    # Read the original dataset
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Define columns to keep (excluding number, date, nozzle_diameter)
    columns_to_keep = [
        'height_1', 'height_2', 'height_3',
        'width_1', 'width_2', 'width_3', 
        'temp', 'humidity',
        'layer_count',  # Keep this as specifically requested
        'slicer_layer_height', 'slicer_layer_width', 
        'slicer_nozzle_speed', 'slicer_extrusion_multiplier'
    ]
    
    # Select only the columns we want to keep
    df_filtered = df[columns_to_keep].copy()
    
    print(f"Filtered dataset shape: {df_filtered.shape}")
    print(f"Columns after filtering: {list(df_filtered.columns)}")
    
    # Convert European decimal notation (commas) to standard notation (dots)
    numeric_columns = [
        'height_1', 'height_2', 'height_3',
        'width_1', 'width_2', 'width_3',
        'temp', 'humidity',
        'slicer_layer_height', 'slicer_layer_width',
        'slicer_nozzle_speed', 'slicer_extrusion_multiplier'
    ]
    
    print("Converting European decimal notation...")
    for col in numeric_columns:
        if col in df_filtered.columns:
            # Convert European decimal format to standard format
            df_filtered[col] = df_filtered[col].apply(convert_european_decimal)
            # Convert to float
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    # Ensure layer_count is integer
    df_filtered['layer_count'] = df_filtered['layer_count'].astype(int)
    
    print(f"Converted dataset shape: {df_filtered.shape}")
    print("Data types after conversion:")
    print(df_filtered.dtypes)
    
    print("\nFirst 5 rows of converted data:")
    print(df_filtered.head())
    
    print("\nSummary statistics:")
    print(df_filtered.describe())
    
    # Check for any missing values
    missing_values = df_filtered.isnull().sum()
    if missing_values.sum() > 0:
        print("\nWarning: Missing values detected:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values detected.")
    
    # Save the converted dataset
    df_filtered.to_csv(output_file, index=False)
    print(f"\nConverted dataset saved to: {output_file}")
    
    return df_filtered

if __name__ == "__main__":
    try:
        converted_df = convert_dataset()
        print("\nConversion completed successfully!")
        
        # Show final column order for comparison with cleaned_df.csv
        print("\nFinal column order:")
        for i, col in enumerate(converted_df.columns, 1):
            print(f"{i:2d}. {col}")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
