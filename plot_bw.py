#!/usr/bin/env python3

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot bandwidth usage from multiple CSV files, separating upload and download.")
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='.',
        help='Directory containing the CSV files (default: current directory)'
    )
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='*.csv',
        help='File pattern to match CSV files (default: *.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='bandwidth_plot',
        help='Output file name prefix for the plots (default: bandwidth_plot)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='Bandwidth Usage Over Time',
        help='Title of the plot'
    )
    return parser.parse_args()

def read_csv_files(directory, pattern):
    # Create the full path pattern
    path_pattern = os.path.join(directory, pattern)
    files = glob.glob(path_pattern)
    if not files:
        print(f"No CSV files found in {directory} matching pattern '{pattern}'.")
        exit(1)
    data_frames = {}
    for file in files:
        try:
            df = pd.read_csv(file)
            # Ensure required columns are present
            if not {'rx_gbps', 'tx_gbps'}.issubset(df.columns):
                print(f"Warning: File '{file}' does not contain required columns 'rx_gbps' and 'tx_gbps'. Skipping.")
                continue
            # Reset index to use row number regardless of any timestamp
            df.reset_index(drop=True, inplace=True)
            data_frames[os.path.basename(file)] = df
        except Exception as e:
            print(f"Error reading '{file}': {e}. Skipping.")
    if not data_frames:
        print("No valid CSV files to process.")
        exit(1)
    return data_frames

def plot_data(data_frames, args):
    # Plot RX (download) bandwidth
    plt.figure(figsize=(15, 8))
    for filename, df in data_frames.items():
        plt.plot(df.index, df['rx_gbps'],  linestyle='-', label=f"{filename} Download (RX)")
    plt.xlabel('Row Index')
    plt.ylabel('Download Bandwidth (Gbps)')
    plt.title(f"{args.title} - Download (RX)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.output}_rx.png")
    tikzplotlib.save('20.tex')
    
    print(f"RX Plot saved as '{args.output}_rx.png'.")
    plt.show()

    # Plot TX (upload) bandwidth
    plt.figure(figsize=(15, 8))
    for filename, df in data_frames.items():
        plt.plot(df.index, df['tx_gbps'],  linestyle='-', label=f"{filename} Upload (TX)")
    plt.xlabel('Row Index')
    plt.ylabel('Upload Bandwidth (Gbps)')
    plt.title(f"{args.title} - Upload (TX)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.output}_tx.png")
    print(f"TX Plot saved as '{args.output}_tx.png'.")
    plt.show()

def main():
    args = parse_arguments()
    data_frames = read_csv_files(args.directory, args.pattern)
    plot_data(data_frames, args)

if __name__ == "__main__":
    main()
