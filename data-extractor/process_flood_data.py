"""
Main script to orchestrate the flood data processing pipeline.
This script runs both enrichment and transformation steps in sequence.
"""

from pathlib import Path
from flood_data_enricher import process_flood_data
from flood_data_enricher_2 import process_flood_data_2

def main():
    try:
        # Get the project root directory (two levels up from this script)
        project_root = Path(__file__).parent.parent

        raw_data_path = project_root / 'data' / 'Flood-Data.csv'
        enriched_data_path = project_root / 'data' / 'Flood-Data-Enriched.csv'
        transformed_data_path = project_root / 'data' / 'Flood-Data-Transformed.csv'
        
        # Step 1: Enrich the raw flood data
        print("Starting data enrichment process...")
        process_flood_data(raw_data_path, enriched_data_path)
        print(f"Data enrichment completed successfully. Output saved to: {enriched_data_path}")
        
        # Step 2: Transform the enriched data
        print("\nStarting data transformation process...")
        process_flood_data_2(enriched_data_path, transformed_data_path)
        print(f"Data transformation completed successfully. Output saved to: {transformed_data_path}")
        
        print("\nComplete data processing pipeline finished successfully!")
        
    except Exception as e:
        print(f"\nError in data processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()