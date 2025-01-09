#!/bin/bash

# Configuration
OUTPUT_DIR="collected_data"
PYTHON_SCRIPT="src/collect_data.py"  # The Python script we created earlier
LOG_FILE="collection_log.txt"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Initialize log file
echo "Data Collection Started at $(date)" > "$LOG_FILE"

# Function to collect data for a specific town
collect_town_data() {
    local town=$1
    local town_dir="$OUTPUT_DIR/$town"
    
    echo "Starting collection for $town at $(date)" | tee -a "$LOG_FILE"
    
    # Create town-specific directory
    mkdir -p "$town_dir"
    
    # Run the Python script with town argument
    python3 "$PYTHON_SCRIPT" "$town" 2>&1 | tee -a "$LOG_FILE"
    
    # Check if collection was successful
    if [ $? -eq 0 ]; then
        echo "Successfully collected data for $town" | tee -a "$LOG_FILE"
    else
        echo "Error collecting data for $town" | tee -a "$LOG_FILE"
    fi
}

# Main execution
echo "Starting data collection across all towns..."

# Loop through towns
for town_num in {01..06}; do
    town="Town${town_num}"
    
    # Run collection for current town
    collect_town_data "$town"
    
    # Add a small delay between towns to ensure clean separation
    sleep 5
done

# Summarize results
echo -e "\nCollection Summary:"
echo "===================="
for town_num in {01..06}; do
    town="Town${town_num}"
    if [ -d "$OUTPUT_DIR/$town" ]; then
        file_count=$(find "$OUTPUT_DIR/$town" -type f | wc -l)
        echo "$town: $file_count files collected"
    else
        echo "$town: No data collected"
    fi
done | tee -a "$LOG_FILE"

echo -e "\nData collection completed at $(date)" | tee -a "$LOG_FILE"

# Check for any towns with no data
missing_data=false
for town_num in {01..06}; do
    town="Town${town_num}"
    if [ ! -d "$OUTPUT_DIR/$town" ] || [ -z "$(ls -A $OUTPUT_DIR/$town 2>/dev/null)" ]; then
        echo "WARNING: No data collected for $town" | tee -a "$LOG_FILE"
        missing_data=true
    fi
done

if [ "$missing_data" = true ]; then
    echo "Some towns have missing data. Check the log file for details."
    exit 1
else
    echo "All towns processed successfully!"
    exit 0
fi