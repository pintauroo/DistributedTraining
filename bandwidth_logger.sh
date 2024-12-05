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

# Get initial counters
rx_bytes_old=$(cat /sys/class/net/"$INTERFACE"/statistics/rx_bytes)
tx_bytes_old=$(cat /sys/class/net/"$INTERFACE"/statistics/tx_bytes)
time_old=$(date +%s.%N)

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

    # Wait for the specified interval
    sleep "$INTERVAL"

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Get current counters
    rx_bytes_new=$(cat /sys/class/net/"$INTERFACE"/statistics/rx_bytes)
    tx_bytes_new=$(cat /sys/class/net/"$INTERFACE"/statistics/tx_bytes)
    time_new=$(date +%s.%N)

    # Calculate time difference
    time_diff=$(echo "$time_new - $time_old" | bc)

    # Calculate bytes per second
    rx_bps=$(echo "($rx_bytes_new - $rx_bytes_old) / $time_diff" | bc -l)
    tx_bps=$(echo "($tx_bytes_new - $tx_bytes_old) / $time_diff" | bc -l)

    # Convert bytes per second to Gbps (bytes * 8 to get bits)
    rx_gbps=$(echo "$rx_bps * 8 / 1000000000" | bc -l)
    tx_gbps=$(echo "$tx_bps * 8 / 1000000000" | bc -l)

    # Format to 6 decimal places
    rx_gbps=$(printf "%.6f" "$rx_gbps")
    tx_gbps=$(printf "%.6f" "$tx_gbps")

    echo "$TIMESTAMP,$rx_gbps,$tx_gbps" | sudo tee -a "$OUTPUT_FILE" > /dev/null

    # Update old values
    rx_bytes_old=$rx_bytes_new
    tx_bytes_old=$tx_bytes_new
    time_old=$time_new
done
