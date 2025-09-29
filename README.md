# Enhanced Intraday Shape-Based Clustering for Gold Price Data

This repository provides a production-ready Python script for performing shape-based clustering on intraday financial time-series data. The primary goal is to identify and analyze recurring daily patterns in gold prices using the k-Shape clustering algorithm.

## Overview

The `intraday_clustering_enhanced.py` script implements a full pipeline for time-series clustering, including:
- **Data Loading and Validation**: Loads a matrix of daily hourly prices, validates data quality, and filters for days with sufficient data.
- **Data Preparation**: Imputes missing values and can operate on either price levels or log returns.
- **Normalization**: Applies appropriate normalization techniques (per-series z-normalization for levels, dataset-wide scaling for returns).
- **Clustering**: Uses `tslearn`'s `KShape` to cluster time series based on shape. It also includes an option to compare results with `TimeSeriesKMeans` using Dynamic Time Warping (DTW).
- **Optimal K-Selection**: Automatically determines the best number of clusters (k) using silhouette scores.
- **Interpretation**: Generates human-readable names for each cluster pattern (e.g., "Strong Bullish Early Peak").
- **Visualization**: Creates a comprehensive multi-plot visualization of the results.
- **Saving Results**: Saves the trained model, cluster assignments, and summary reports.

## Getting Started

### Prerequisites

- Python 3.7+
- The script requires a data file named `daily_hourly_close_matrix.csv` in the root directory. This CSV should have a 'day' column as the index and hourly price columns.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required packages:**
    The necessary Python libraries are listed at the top of the script. You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn tslearn scikit-learn joblib
    ```

## How to Run the Analysis

To run the clustering pipeline, execute the script from your terminal:

```bash
python intraday_clustering_enhanced.py
```

The script will print its progress to the console, from loading data to saving the final results and generating plots.

### Configuration

You can customize the analysis by modifying the constants in the `CONFIGURATION` section of the script:
- `INPUT_MATRIX_CSV`: Path to the input data file.
- `OUTPUT_DIR`: Directory to save results.
- `K_RANGE`: The range of `k` values to test for clustering.
- `USE_RETURNS`: Set to `True` to cluster based on log returns instead of price levels.
- `SAMPLE_SIZE`: The number of days to include in the analysis.

## Output Files

The script will create an `clustering_results` directory (or the directory specified in `OUTPUT_DIR`) with the following files:

- **`enhanced_clustering_analysis.png`**: A multi-panel plot that provides a comprehensive overview of the clustering results, including:
    - K-selection silhouette scores.
    - Cluster centroids.
    - Sample patterns from each cluster.
    - Cluster size distribution.
    - Silhouette analysis plot.
    - Temporal distribution of clusters.

- **`kshape_model_enhanced.pkl`**: A `joblib` file containing a dictionary of key artifacts, including:
    - The trained clustering model (`model`).
    - The data scaler (`scaler`).
    - The final cluster assignments (`labels`).
    - Other metadata and results for reproducibility.

- **`enhanced_cluster_assignments.csv`**: A CSV file that maps each date to its assigned cluster. It includes the pattern name and the silhouette score for each day, providing a measure of confidence.

- **`cluster_summary.csv`**: A CSV file containing summary statistics for each cluster, including its size, percentage of the dataset, and average silhouette score.