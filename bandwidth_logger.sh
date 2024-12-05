#!/bin/bash

# bandwidth_logger.sh
# Logs bandwidth usage of a specified network interface to a CSV file in Gbps every second.

# Function to display usage information
usage() {
    echo "Usage: $0 [-i INTERFACE] [-o OUTPUT_FILE] [-d DURATION] [-h]"
    echo ""
    echo "Options:"
    echo "  -i, --interface    Network interface to monitor (default: enp7s0)"
    echo "  -o, --output       Output CSV file path (default: /var/log/bandwidth_usage.csv)"
    echo "  -d, --duration     Total recording time in seconds (default: run indefinitely)"
    echo "  -h, --help         Display this help message and exit"
}

# Default configuration
INTERFACE="enp7s0"
OUTPUT_FILE="/var/log/bandwidth_usage.csv"
INTERVAL=1          # Interval in seconds between measurements
DURATION=0          # Total recording time in seconds (0 means run indefinitely)

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--interface)
            INTERFACE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            exit 1
            ;;
    esac
done

# Function to check if the interface exists
check_interface() {
    if ! ip link show "$INTERFACE" &> /dev/null; then
        echo "Error: Network interface '$INTERFACE' does not exist."
        exit 1
    fi
}

# Function to ensure the output directory exists and is writable
prepare_output_file() {
    OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Creating directory: $OUTPUT_DIR"
        sudo mkdir -p "$OUTPUT_DIR"
        sudo chown "$(whoami)": "$OUTPUT_DIR"
    fi

    if [ ! -w "$OUTPUT_DIR" ]; then
        echo "Error: Directory '$OUTPUT_DIR' is not writable."
        exit 1
    fi
}

# Check if the specified interface exists
check_interface

# Prepare the output file
prepare_output_file

# Initialize CSV file with headers if it doesn't exist
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "timestamp,rx_gbps,tx_gbps" | sudo tee "$OUTPUT_FILE" > /dev/null
fi

# Calculate the end time if duration is specified
if [ "$DURATION" -gt 0 ]; then
    START_TIME=$(date +%s)
    END_TIME=$((START_TIME + DURATION))
fi

# Infinite loop to log bandwidth usage
while true; do
    # If duration is set and current time exceeds end time, exit
    if [ "$DURATION" -gt 0 ]; then
        CURRENT_TIME=$(date +%s)
        if [ "$CURRENT_TIME" -ge "$END_TIME" ]; then
            echo "Recording duration of $DURATION seconds completed. Exiting."
            exit 0
        fi
    fi

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Capture bandwidth usage using ifstat
    # The -i flag specifies the interface, the 1 1 at the end sets a single interval of 1 second
    # The -b flag outputs bytes per second; using -k for Kbps
    # Adjust according to your ifstat version and preferred units
    OUTPUT=$(ifstat -i "$INTERFACE" 1 1 | awk 'NR==3 {print $1","$2}')

    # Check if OUTPUT is not empty
    if [ -n "$OUTPUT" ]; then
        # Convert Kbps to Gbps by dividing by 1,000,000
        RX_KBPS=$(echo "$OUTPUT" | cut -d',' -f1)
        TX_KBPS=$(echo "$OUTPUT" | cut -d',' -f2)

        # Handle cases where ifstat outputs '-' or other non-numeric values
        if [[ "$RX_KBPS" =~ ^[0-9.]+$ ]] && [[ "$TX_KBPS" =~ ^[0-9.]+$ ]]; then
            RX_GBPS=$(awk "BEGIN {printf \"%.6f\", $RX_KBPS/1000000}")
            TX_GBPS=$(awk "BEGIN {printf \"%.6f\", $TX_KBPS/1000000}")
            echo "$TIMESTAMP,$RX_GBPS,$TX_GBPS" | sudo tee -a "$OUTPUT_FILE" > /dev/null
        else
            echo "$TIMESTAMP,ERROR,ERROR" | sudo tee -a "$OUTPUT_FILE" > /dev/null
        fi
    else
        echo "$TIMESTAMP,ERROR,ERROR" | sudo tee -a "$OUTPUT_FILE" > /dev/null
    fi

    # Wait for the specified interval before next measurement
    sleep "$INTERVAL"
done
