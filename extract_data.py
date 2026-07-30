#!/usr/bin/env python3
"""
Script to extract a subset of date-filtered data from a complete stock/index CSV file.
By default, extracts data from start year 2000 to end year 2020.
 uv run python extract_2000_2020.py --input data/reall-complete-SAN-2000-2025.csv --output data/reall-complete-SAN-2000-2020.csv --start-year 2000 --end-year 2020
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def extract_data_by_year_range(
    input_file: Path,
    output_file: Path,
    start_year: int = 2000,
    end_year: int = 2020
) -> None:
    """Filter CSV dataset by date range [start_year, end_year] and save result.

    Args:
        input_file: Path to the source CSV file.
        output_file: Path where filtered CSV file will be written.
        start_year: Inclusive starting year for filtering.
        end_year: Inclusive ending year for filtering.
    """
    logging.info("Reading source dataset from %s ...", input_file)
    df = pd.read_csv(input_file)
    logging.info("Original dataset contains %d rows.", len(df))

    start_date = f"{start_year:04d}-01-01"
    end_date = f"{end_year:04d}-12-31"

    logging.info("Filtering data for Date between %s and %s ...", start_date, end_date)
    
    # Clean and filter dates
    df["Date"] = df["Date"].astype(str).str.strip()
    filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    logging.info("Filtered dataset contains %d rows.", len(filtered_df))

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Saving filtered data to %s ...", output_file)
    filtered_df.to_csv(output_file, index=False)
    logging.info("Successfully saved %d records to %s", len(filtered_df), output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract date-range filtered data from CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/reall-complete-2000-2025.csv"),
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reall-complete-2000-2020.csv"),
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="Start year (inclusive)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2020,
        help="End year (inclusive)"
    )

    args = parser.parse_args()
    extract_data_by_year_range(
        input_file=args.input,
        output_file=args.output,
        start_year=args.start_year,
        end_year=args.end_year
    )


if __name__ == "__main__":
    main()
