#!/bin/bash

# Usage function to display help
usage() {
    echo "Usage: $0 <bandwidth>"
    echo "Example: $0 5gbit"
    exit 1
}

# Check if bandwidth argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Bandwidth size not provided."
    usage
fi

BANDWIDTH=$1

# Validate bandwidth format (simple regex)
if ! [[ $BANDWIDTH =~ ^[0-9]+(kbit|mbit|gbit)$ ]]; then
    echo "Error: Invalid bandwidth format. Use formats like 500kbit, 10mbit, 5gbit."
    usage
fi

INTERFACE="enp7s0"

# Clear existing qdisc
echo "Deleting existing qdisc on $INTERFACE..."
sudo tc qdisc del dev "$INTERFACE" root 2>/dev/null

# Add root qdisc
echo "Adding root qdisc..."
sudo tc qdisc add dev "$INTERFACE" root handle 1: htb default 10

# Create parent class
echo "Creating parent class with rate and ceil set to $BANDWIDTH..."
sudo tc class add dev "$INTERFACE" parent 1: classid 1:1 htb rate "$BANDWIDTH" ceil "$BANDWIDTH"

# Create child class
echo "Creating child class with rate and ceil set to $BANDWIDTH..."
sudo tc class add dev "$INTERFACE" parent 1:1 classid 1:10 htb rate "$BANDWIDTH" ceil "$BANDWIDTH"

# Add filter
echo "Adding filter to match all IP traffic to class 1:10..."
sudo tc filter add dev "$INTERFACE" protocol ip parent 1:0 prio 1 u32 match ip src 0.0.0.0/0 flowid 1:10

echo "Bandwidth setting applied successfully on $INTERFACE with $BANDWIDTH."
