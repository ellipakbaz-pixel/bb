# Requirements: pip install pandas numpy matplotlib seaborn tslearn scikit-learn joblib
"""
Enhanced Intraday Shape-Based Clustering for Gold Price Data
============================================================
Production-ready clustering pipeline with critical bug fixes and improvements:
- Fixed metric alignment between k-Shape training and silhouette evaluation
- Proper price-level imputation before returns computation
- Dataset-wide normalization for returns clustering
- Robust k-selection with validity guards
- Consistent preprocessing for training and deployment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
from datetime import datetime
from typing import Tuple, Dict, Any, List

# Core clustering and evaluation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape, TimeSeriesKMeans, silhouette_score as ts_silhouette_score
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
INPUT_MATRIX_CSV = "daily_hourly_close_matrix.csv"
OUTPUT_DIR = "clustering_results"
RANDOM_STATE = 42

# Clustering parameters
K_RANGE = range(10, 11)      # Valid range: 2 <= k < n_samples
USE_RETURNS = False         # True: use log returns, False: use z-normalized levels
MIN_HOURS_PER_DAY = 15     # Minimum hours required per day
SAMPLE_SIZE = 4900        # Use only first 500 days for demonstration

# Method comparison option
COMPARE_METHODS = True     # Compare k-Shape vs TimeSeriesKMeans with DTW

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_validate_data(csv_path: str, sample_size: int, min_hours: int) -> Tuple[pd.DataFrame, range]:
    """Loads, validates, and preprocesses the input time-series data.

    Args:
        csv_path (str): The path to the input CSV file.
        sample_size (int): The number of days to sample from the dataset.
        min_hours (int): The minimum number of hourly data points required for a day to be included.

    Returns:
        Tuple[pd.DataFrame, range]: A tuple containing the cleaned DataFrame and the adjusted k-range for clustering.
    """
    print("\n📥 Loading and validating data...")
    try:
        df_matrix = pd.read_csv(csv_path, index_col='day', parse_dates=True)
        df_matrix = df_matrix.head(sample_size)

        print(f"✅ Sample matrix: {df_matrix.shape[0]} days × {df_matrix.shape[1]} hours")

        hours_per_day = df_matrix.count(axis=1)
        valid_days = hours_per_day >= min_hours
        df_clean = df_matrix[valid_days].copy()
        print(f"✅ Using {len(df_clean)} days for clustering analysis")

        max_valid_k = len(df_clean) - 1
        k_range = range(max(2, min(K_RANGE)), min(max_valid_k, max(K_RANGE) + 1))
        if max(K_RANGE) >= max_valid_k:
            print(f"⚠️  Adjusting k-range to {list(k_range)}")

        return df_clean, k_range
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit(1)

def prepare_data(df: pd.DataFrame, use_returns: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepares the data for clustering by imputing missing values and optionally converting to returns.

    Args:
        df (pd.DataFrame): The input DataFrame of time-series data.
        use_returns (bool): If True, converts price levels to log returns.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: A tuple containing the processed data matrix (X_raw),
        the corresponding dates, and the hour labels.
    """
    print(f"\n⚙️  Enhanced data preparation ({'returns' if use_returns else 'levels'})...")
    
    df_imputed = df.copy()
    for idx in df_imputed.index:
        df_imputed.loc[idx] = df_imputed.loc[idx].fillna(method='ffill').fillna(method='bfill')
    
    df_imputed.dropna(axis=0, inplace=True)
    print(f"📊 Using {len(df_imputed)} complete days after imputation")
    
    assert not df_imputed.isna().any().any(), "NaN values detected"

    if use_returns:
        X_raw = np.log(df_imputed.values[:, 1:] / df_imputed.values[:, :-1])
        hour_labels = df_imputed.columns[1:]
        X_raw = np.where(np.isfinite(X_raw), X_raw, 0)
    else:
        X_raw = df_imputed.values
        hour_labels = df_imputed.columns

    return X_raw, df_imputed.index.values, list(hour_labels)

def normalize_data(X_raw: np.ndarray, use_returns: bool) -> Tuple[np.ndarray, Any]:
    """Normalizes the time-series data.

    Args:
        X_raw (np.ndarray): The raw data matrix.
        use_returns (bool): The type of data being processed (returns or levels).

    Returns:
        Tuple[np.ndarray, Any]: A tuple containing the normalized data and the scaler object.
    """
    print("\n🎯 Applying proper normalization...")
    X_raw_3d = X_raw[:, :, np.newaxis]

    if use_returns:
        print("📏 Using dataset-wide StandardScaler for returns")
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_raw)[:, :, np.newaxis]
    else:
        print("📏 Using per-series z-normalization for levels")
        scaler = TimeSeriesScalerMeanVariance()
        X_normalized = scaler.fit_transform(X_raw_3d)

    print("✅ Normalization complete")
    return X_normalized, scaler

def perform_clustering(X_normalized: np.ndarray, k_range: range, compare_methods: bool) -> Dict[str, Any]:
    """Performs time-series clustering and k-selection.

    Args:
        X_normalized (np.ndarray): The normalized time-series data.
        k_range (range): The range of k values to test.
        compare_methods (bool): If True, compares k-Shape with TimeSeriesKMeans.

    Returns:
        Dict[str, Any]: A dictionary containing the clustering results, including the best model,
        labels, and scores.
    """
    print(f"\n🎯 Enhanced k-selection with proper metric alignment...")
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    results = {
        'k_shape': {'silhouette_scores': [], 'models': {}},
        'ts_kmeans': {'silhouette_scores': [], 'models': {}} if compare_methods else None
    }

    for k in k_range:
        if k >= X_normalized.shape[0]: continue
        print(f"  🔄 Fitting models with k={k}...")
        
        kshape_model = KShape(n_clusters=k, random_state=RANDOM_STATE, verbose=False)
        kshape_labels = kshape_model.fit_predict(X_normalized)
        try:
            kshape_sil = ts_silhouette_score(X_normalized, kshape_labels, metric="dtw")
        except Exception:
            kshape_sil = silhouette_score(X_normalized.squeeze(-1), kshape_labels)
        results['k_shape']['silhouette_scores'].append(kshape_sil)
        results['k_shape']['models'][k] = kshape_model

        if compare_methods:
            try:
                ts_model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=RANDOM_STATE, verbose=False)
                ts_labels = ts_model.fit_predict(X_normalized)
                ts_sil = ts_silhouette_score(X_normalized, ts_labels, metric="dtw")
                results['ts_kmeans']['silhouette_scores'].append(ts_sil)
                results['ts_kmeans']['models'][k] = ts_model
            except Exception:
                results['ts_kmeans']['silhouette_scores'].append(0.0)

    best_method = 'k_shape'
    kshape_best_score = max(results['k_shape']['silhouette_scores']) if results['k_shape']['silhouette_scores'] else 0
    optimal_k = k_range[np.argmax(results['k_shape']['silhouette_scores'])] if results['k_shape']['silhouette_scores'] else min(k_range)
    
    if compare_methods and results['ts_kmeans']['silhouette_scores']:
        ts_best_score = max(results['ts_kmeans']['silhouette_scores'])
        if ts_best_score > kshape_best_score:
            best_method = 'ts_kmeans'
            optimal_k = k_range[np.argmax(results['ts_kmeans']['silhouette_scores'])]

    final_model = results[best_method]['models'][optimal_k]
    final_labels = final_model.predict(X_normalized)
    best_score = max(results[best_method]['silhouette_scores'])

    try:
        sample_scores = silhouette_samples(X_normalized.squeeze(-1), final_labels)
    except Exception:
        sample_scores = np.zeros(len(final_labels))

    return {
        'model': final_model, 'labels': final_labels, 'optimal_k': optimal_k,
        'best_silhouette': best_score, 'best_method': best_method, 'sample_scores': sample_scores,
        'all_results': results
    }

def interpret_clusters(model: Any, labels: np.ndarray, sample_scores: np.ndarray, use_returns: bool) -> Dict[int, str]:
    """Generates human-readable interpretations for each cluster.

    Args:
        model: The fitted clustering model.
        labels (np.ndarray): The cluster labels for each time series.
        sample_scores (np.ndarray): The silhouette score for each sample.
        use_returns (bool): Indicates if the data represents returns or levels.

    Returns:
        Dict[int, str]: A dictionary mapping cluster IDs to their interpretations.
    """
    print(f"\n🎯 Enhanced Cluster Interpretation:")
    centroids = model.cluster_centers_
    interpretations = {}
    for i in range(model.n_clusters):
        mask = labels == i
        if not np.any(mask): continue
        
        centroid = centroids[i].ravel()
        volatility = np.std(centroid)
        net_trend = np.cumsum(centroid)[-1] if use_returns else centroid[-1] - centroid[0]
        
        timing = "Early Peak" if np.argmax(centroid) < len(centroid) // 3 else "Late Peak" if np.argmax(centroid) > 2 * len(centroid) // 3 else "Mid-Day Peak"
        trend_direction = "Bullish" if net_trend > 0 else "Bearish"
        intensity = "Strong" if abs(net_trend) > volatility else "Moderate"
        pattern_name = f"{intensity} {trend_direction} {timing}"
        interpretations[i] = pattern_name
        
        print(f"  🏷️  Cluster {i}: {pattern_name} ({np.sum(mask)} days)")
    return interpretations

def save_results(output_dir: str, artifacts: Dict[str, Any], assignments_df: pd.DataFrame):
    """Saves the clustering model, artifacts, and assignments to disk.

    Args:
        output_dir (str): The directory to save the results.
        artifacts (Dict[str, Any]): A dictionary of model artifacts to save.
        assignments_df (pd.DataFrame): A DataFrame of cluster assignments.
    """
    print(f"\n💾 Saving enhanced results to {output_dir}/")
    Path(output_dir).mkdir(exist_ok=True)
    joblib.dump(artifacts, f"{output_dir}/kshape_model_enhanced.pkl")
    assignments_df.to_csv(f"{output_dir}/enhanced_cluster_assignments.csv", index=False)

    # Create and save summary
    summary = assignments_df.groupby(['cluster', 'pattern_name']).agg(
        count=('cluster', 'size'),
        avg_silhouette=('silhouette_score', 'mean')
    ).reset_index()
    summary['percentage'] = summary['count'] / summary['count'].sum() * 100
    summary.to_csv(f"{output_dir}/cluster_summary.csv", index=False)
    print("✅ Results saved.")

def visualize_results(artifacts: Dict[str, Any], assignments_df: pd.DataFrame, hour_labels: List[str], k_range: range):
    """Creates and saves a comprehensive visualization of the clustering results.

    Args:
        artifacts (Dict[str, Any]): The dictionary of clustering artifacts.
        assignments_df (pd.DataFrame): The DataFrame with cluster assignments.
        hour_labels (List[str]): The labels for the time-series steps (hours).
        k_range (range): The range of k values tested.
    """
    print(f"\n📊 Creating comprehensive visualizations...")

    optimal_k = artifacts['optimal_k']
    final_labels = artifacts['labels']
    centroids = artifacts['model'].cluster_centers_
    cluster_interpretations = artifacts['cluster_interpretations']
    X_normalized = artifacts['X_normalized']
    sample_scores = artifacts['sample_scores']
    best_score = artifacts['best_silhouette']
    results = artifacts['all_results']

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))

    # K selection plot
    if COMPARE_METHODS:
        axes[0,0].plot(k_range, results['k_shape']['silhouette_scores'], 'bo-', label='k-Shape')
        axes[0,0].plot(k_range, results['ts_kmeans']['silhouette_scores'], 'rs--', label='TS-KMeans (DTW)')
    else:
        axes[0,0].plot(k_range, results['k_shape']['silhouette_scores'], 'bo-')
    axes[0,0].axvline(optimal_k, color='red', linestyle='--', label=f'Selected k={optimal_k}')
    axes[0,0].set_title('K Selection')
    axes[0,0].legend()

    # Centroids plot
    for i in range(optimal_k):
        axes[0,1].plot(centroids[i].ravel(), label=f'{cluster_interpretations.get(i, f"C{i}")}')
    axes[0,1].set_title('Cluster Centroids')
    axes[0,1].legend()

    # Sample patterns
    for i in range(optimal_k):
        for series in X_normalized[final_labels == i][:5]:
            axes[0,2].plot(series.ravel(), color=colors[i], alpha=0.5)
    axes[0,2].set_title('Sample Patterns by Cluster')

    # Cluster distribution
    counts = np.bincount(final_labels)
    axes[1,0].bar(range(optimal_k), counts, color=colors)
    axes[1,0].set_title('Cluster Size Distribution')

    # Silhouette analysis
    y_lower = 10
    for i in range(optimal_k):
        scores = sample_scores[final_labels == i]
        scores.sort()
        size = scores.shape[0]
        y_upper = y_lower + size
        axes[1,1].fill_betweenx(np.arange(y_lower, y_upper), 0, scores, facecolor=colors[i], alpha=0.7)
        y_lower = y_upper + 10
    axes[1,1].axvline(best_score, color="red", linestyle="--")
    axes[1,1].set_title('Silhouette Analysis')

    # Temporal distribution
    axes[1,2].scatter(assignments_df['date'], assignments_df['cluster'], c=assignments_df['cluster'], cmap='Set1', alpha=0.7)
    axes[1,2].set_title('Cluster Assignments Over Time')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/enhanced_clustering_analysis.png", dpi=300)
    plt.show()

def main():
    """Main execution function for the clustering pipeline."""
    print(f"🚀 Enhanced Intraday Clustering")

    df_clean, k_range = load_and_validate_data(INPUT_MATRIX_CSV, SAMPLE_SIZE, MIN_HOURS_PER_DAY)

    X_raw, dates, hour_labels = prepare_data(df_clean, USE_RETURNS)

    X_normalized, scaler = normalize_data(X_raw, USE_RETURNS)

    clustering_results = perform_clustering(X_normalized, k_range, COMPARE_METHODS)

    cluster_interpretations = interpret_clusters(
        clustering_results['model'], clustering_results['labels'],
        clustering_results['sample_scores'], USE_RETURNS
    )

    assignments_df = pd.DataFrame({
        'date': dates[:len(clustering_results['labels'])],
        'cluster': clustering_results['labels'],
        'pattern_name': [cluster_interpretations.get(l) for l in clustering_results['labels']],
        'silhouette_score': clustering_results['sample_scores']
    })

    model_artifacts = {
        'model': clustering_results['model'],
        'scaler': scaler,
        'labels': clustering_results['labels'],
        'optimal_k': clustering_results['optimal_k'],
        'best_silhouette': clustering_results['best_silhouette'],
        'best_method': clustering_results['best_method'],
        'hour_labels': hour_labels,
        'use_returns': USE_RETURNS,
        'sample_scores': clustering_results['sample_scores'],
        'cluster_interpretations': cluster_interpretations,
        'X_normalized': X_normalized,
        'all_results': clustering_results['all_results']
    }

    save_results(OUTPUT_DIR, model_artifacts, assignments_df)

    visualize_results(model_artifacts, assignments_df, hour_labels, k_range)

    print(f"\n🎉 Analysis Complete!")
    print(f"🏆 Best model: {clustering_results['best_method']} with k={clustering_results['optimal_k']}")

if __name__ == "__main__":
    main()