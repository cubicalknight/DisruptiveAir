#!/bin/bash

# Usage: ./extract_data.sh <years> <months>
# Example: ./extract_data.sh "2020 2021 2022" "1 2 12"
# Example extracts data for months January, February, and December for each year 2020, 2021, and 2022

YEARS=$1
MONTHS=$2

BASE_DIR=$(dirname "$0")/../data/ot_reporting

if [[ -z "$YEARS" || -z "$MONTHS" ]]; then
    echo "Usage: $0 <years> <months>"
    exit 1
fi

mkdir -p "$BASE_DIR"

for YEAR in $YEARS; do
    for MONTH in $MONTHS; do
        # Construct the download URL based on user input
        BASE_URL="https://transtats.bts.gov/PREZIP"
        ZIP_FILE="On_Time_Reporting_Carrier_On_Time_Performance_1987_present_${YEAR}_${MONTH}.zip"
        URL="$BASE_URL/$ZIP_FILE"

        # Make a temporary directory to download the file
        TEMP_DIR=$(mktemp -d)

        # Download the zip file
        echo "Downloading $URL..."
        curl -o "$TEMP_DIR/$ZIP_FILE" -L -k "$URL"

        # Check if the file was downloaded successfully
        if [[ $? -ne 0 || ! -f "$TEMP_DIR/$ZIP_FILE" ]]; then
            echo "Failed to download $URL"
            rm -rf "$TEMP_DIR"
            continue
        fi

        # Filter for only CSV files and extract them
        echo "Extracting CSV files from $ZIP_FILE..."
        unzip -j "$TEMP_DIR/$ZIP_FILE" "*.csv" -d "$BASE_DIR"

        # Remove the temporary directory
        rm -rf "$TEMP_DIR"

        echo "CSV files from $YEAR-$MONTH extracted to $BASE_DIR"
    done
done
