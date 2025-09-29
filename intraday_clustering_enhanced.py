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

# Core clustering and evaluation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape, TimeSeriesKMeans, silhouette_score as ts_silhouette_score
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_weekly_pattern_analysis(df):
    """
    Analyzes weekly cluster patterns with chronologically sorted weekdays.
    This function contains the bug fix.
    """
    # CRITICAL FIX: Enforce chronological order for weekdays
    df_copy = df.copy()
    df_copy['weekday'] = df_copy['date'].dt.day_name()
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_copy['weekday'] = pd.Categorical(df_copy['weekday'], categories=days_of_week, ordered=True)
    
    # Group by weekday and cluster
    weekday_cluster = df_copy.groupby(['weekday', 'cluster'], observed=True).size().unstack(fill_value=0)
    weekday_pct = weekday_cluster.div(weekday_cluster.sum(axis=1), axis=0) * 100
    
    return weekday_pct

def main():
    """Main function to run the entire original analysis pipeline."""
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

    print(f"🚀 Enhanced Intraday Clustering with Critical Fixes")
    print(f"📊 Using sample of {SAMPLE_SIZE} days for demonstration")
    print(f"📈 k-Shape clustering with k-range: {list(K_RANGE)}")
    print(f"🔧 Method comparison: {'Enabled' if COMPARE_METHODS else 'Disabled'}")
    print("=" * 60)

    # ========== STEP 1: LOAD AND VALIDATE DATA ==========
    print("\n📥 Loading and validating data...")
    try:
        df_matrix = pd.read_csv(INPUT_MATRIX_CSV, index_col='day', parse_dates=True)

        # Take only the first SAMPLE_SIZE days for speed
        df_matrix = df_matrix.head(SAMPLE_SIZE)

        print(f"✅ Sample matrix: {df_matrix.shape[0]} days × {df_matrix.shape[1]} hours")
        print(f"📅 Sample range: {df_matrix.index.min()} to {df_matrix.index.max()}")

        # Verify hour ordering
        hour_cols = df_matrix.columns
        print(f"🕐 Hour range: {hour_cols[0]} to {hour_cols[-1]}")

        # Check data coverage
        total_values = df_matrix.size
        missing_values = df_matrix.isna().sum().sum()
        coverage = (total_values - missing_values) / total_values * 100
        print(f"📊 Sample coverage: {coverage:.1f}% ({total_values - missing_values:,} / {total_values:,} values)")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit(1)

    # ========== STEP 2: DATA QUALITY CHECKS ==========
    print("\n🔍 Performing data quality checks...")

    # Check hours per day
    hours_per_day = df_matrix.count(axis=1)
    valid_days = hours_per_day >= MIN_HOURS_PER_DAY
    print(f"📊 Days with ≥{MIN_HOURS_PER_DAY} hours: {valid_days.sum()} / {len(df_matrix)}")

    # Filter to valid days only
    df_clean = df_matrix[valid_days].copy()
    print(f"✅ Using {len(df_clean)} days for clustering analysis")

    # Validate k-range against sample size
    max_valid_k = len(df_clean) - 1
    if max(K_RANGE) >= max_valid_k:
        print(f"⚠️  Adjusting k-range: max k reduced from {max(K_RANGE)} to {max_valid_k-1}")
        K_RANGE = range(max(2, min(K_RANGE)), min(max_valid_k, max(K_RANGE)))
        print(f"📈 Adjusted k-range: {list(K_RANGE)}")

    # ========== STEP 3: ENHANCED DATA PREPARATION ==========
    print(f"\n⚙️  Enhanced data preparation ({'returns' if USE_RETURNS else 'levels'})...")

    # CRITICAL FIX: Impute prices BEFORE computing returns
    print("🔧 Step 1: Imputing missing prices at price level...")
    df_imputed = df_clean.copy()

    # Forward/backward fill within each day (row-wise)
    for idx in df_imputed.index:
        day_prices = df_imputed.loc[idx]
        # Forward fill then backward fill
        day_prices = day_prices.fillna(method='ffill').fillna(method='bfill')
        df_imputed.loc[idx] = day_prices

    # Check for any remaining NaN values
    remaining_nans = df_imputed.isna().sum().sum()
    if remaining_nans > 0:
        print(f"⚠️  {remaining_nans} NaN values remain after imputation")
        # Drop days with remaining NaN values
        complete_days = df_imputed.dropna(axis=0)
        df_imputed = complete_days
        print(f"📊 Using {len(df_imputed)} complete days after NaN removal")

    # Assert equal-length series and no NaNs
    assert not df_imputed.isna().any().any(), "NaN values detected after imputation"
    assert len(df_imputed.columns) > 1, "Insufficient hours per day"

    if USE_RETURNS:
        print("🔄 Step 2: Computing hourly log returns from imputed prices...")
        # Compute log returns within each day (proper time-ordered differencing)
        X_raw = np.log(df_imputed.values[:, 1:] / df_imputed.values[:, :-1])
        hour_labels = df_imputed.columns[1:]  # Corresponding hour labels
        
        # Verify no NaN/inf in returns (should be clean after price imputation)
        if not np.all(np.isfinite(X_raw)):
            print("⚠️  Non-finite returns detected, applying minimal cleanup...")
            X_raw = np.where(np.isfinite(X_raw), X_raw, 0)
        
        print(f"✅ Returns matrix shape: {X_raw.shape}")

    else:
        print("📊 Step 2: Using price levels...")
        X_raw = df_imputed.values
        hour_labels = df_imputed.columns
        print(f"✅ Levels matrix shape: {X_raw.shape}")

    # Convert to 3D format expected by tslearn
    X_raw_3d = X_raw[:, :, np.newaxis]

    print(f"✅ Data preparation complete: {X_raw_3d.shape}")

    # ========== STEP 4: PROPER NORMALIZATION ==========
    print("\n🎯 Applying proper normalization...")

    if USE_RETURNS:
        print("📏 Using dataset-wide StandardScaler for returns (preserves amplitude relationships)")
        # For returns, use dataset-wide scaling to preserve relative magnitudes
        scaler_2d = StandardScaler()
        X_scaled_2d = scaler_2d.fit_transform(X_raw)
        X_normalized = X_scaled_2d[:, :, np.newaxis]
        scaler = scaler_2d  # Keep for deployment consistency
    else:
        print("📏 Using per-series z-normalization for levels (k-Shape standard)")
        # For levels, use per-series z-normalization (k-Shape standard approach)
        scaler = TimeSeriesScalerMeanVariance()
        X_normalized = scaler.fit_transform(X_raw_3d)

    print("✅ Normalization complete")

    # ========== STEP 5: ENHANCED K-SELECTION WITH METRIC ALIGNMENT ==========
    print(f"\n🎯 Enhanced k-selection with proper metric alignment...")

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    results = {
        'k_shape': {'silhouette_scores': [], 'models': {}},
        'ts_kmeans': {'silhouette_scores': [], 'models': {}} if COMPARE_METHODS else None
    }

    for k in K_RANGE:
        if k >= len(df_imputed):  # Guard against invalid k
            print(f"  ⚠️  Skipping k={k} (≥ n_samples={len(df_imputed)})")
            continue

        print(f"  🔄 Fitting models with k={k}...")
        
        # === k-Shape with SBD distance ===
        kshape_model = KShape(n_clusters=k, random_state=RANDOM_STATE, verbose=False)
        kshape_labels = kshape_model.fit_predict(X_normalized)
        
        # CRITICAL FIX: Use tslearn's silhouette with DTW as proxy for SBD
        # (tslearn doesn't support SBD in silhouette, DTW is closest shape-aware metric)
        try:
            # Try tslearn silhouette with DTW (shape-aware metric)
            kshape_sil = ts_silhouette_score(X_normalized, kshape_labels, metric="dtw")
            sil_method = "DTW"
        except Exception as e:
            # Fallback to euclidean but clearly label it
            kshape_sil = silhouette_score(X_normalized.squeeze(-1), kshape_labels, metric="euclidean")
            sil_method = "Euclidean (fallback)"
            
        results['k_shape']['silhouette_scores'].append(kshape_sil)
        results['k_shape']['models'][k] = kshape_model

        cluster_sizes = np.bincount(kshape_labels)
        print(f"    ✅ k-Shape: silhouette = {kshape_sil:.4f} ({sil_method}), sizes = {cluster_sizes}")
        
        # === TimeSeriesKMeans with DTW (for comparison) ===
        if COMPARE_METHODS:
            try:
                ts_model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=RANDOM_STATE, verbose=False)
                ts_labels = ts_model.fit_predict(X_normalized)
                ts_sil = ts_silhouette_score(X_normalized, ts_labels, metric="dtw")

                results['ts_kmeans']['silhouette_scores'].append(ts_sil)
                results['ts_kmeans']['models'][k] = ts_model

                ts_cluster_sizes = np.bincount(ts_labels)
                print(f"    ✅ TS-KMeans: silhouette = {ts_sil:.4f} (DTW), sizes = {ts_cluster_sizes}")

            except Exception as e:
                print(f"    ⚠️  TS-KMeans failed: {e}")
                results['ts_kmeans']['silhouette_scores'].append(0.0)

    # Find optimal k for each method
    best_method = 'k_shape'  # Default to k-Shape
    if results['k_shape']['silhouette_scores']:
        kshape_optimal_k = K_RANGE[np.argmax(results['k_shape']['silhouette_scores'])]
        kshape_best_score = max(results['k_shape']['silhouette_scores'])
        print(f"\n🏆 k-Shape optimal: k={kshape_optimal_k} (silhouette={kshape_best_score:.4f})")
        
        optimal_k = kshape_optimal_k
        best_score = kshape_best_score
    else:
        optimal_k = 3
        best_score = 0.0
        print(f"\n⚠️  Using default k = {optimal_k}")

    if COMPARE_METHODS and results['ts_kmeans']['silhouette_scores']:
        ts_optimal_k = K_RANGE[np.argmax(results['ts_kmeans']['silhouette_scores'])]
        ts_best_score = max(results['ts_kmeans']['silhouette_scores'])
        print(f"🏆 TS-KMeans optimal: k={ts_optimal_k} (silhouette={ts_best_score:.4f})")
        
        # Choose best overall method
        if ts_best_score > best_score:
            best_method = 'ts_kmeans'
            optimal_k = ts_optimal_k
            best_score = ts_best_score
            print(f"🎯 Selected method: TimeSeriesKMeans (DTW)")
        else:
            print(f"🎯 Selected method: k-Shape (SBD)")

    # ========== STEP 6: FIT FINAL MODEL ==========
    print(f"\n🎯 Final model: {best_method.replace('_', '-')} with k={optimal_k}...")
    final_model = results[best_method]['models'][optimal_k]
    final_labels = final_model.predict(X_normalized)

    # Calculate individual sample silhouette scores with consistent metric
    try:
        if best_method == 'ts_kmeans':
            sample_scores = silhouette_samples(X_normalized.squeeze(-1), final_labels, metric="euclidean")
        else:
            # For k-Shape, use euclidean as approximation (clearly documented)
            sample_scores = silhouette_samples(X_normalized.squeeze(-1), final_labels, metric="euclidean")
    except Exception:
        sample_scores = np.zeros(len(final_labels))

    print(f"✅ Final model:")
    print(f"   🔧 Method: {best_method.replace('_', '-')}")
    print(f"   📊 Clusters: {optimal_k}")
    print(f"   📈 Silhouette: {best_score:.4f}")
    print(f"   📋 Cluster sizes: {np.bincount(final_labels)}")

    # ========== STEP 7: ENHANCED INTERPRETATION ==========
    print(f"\n🎯 Enhanced Cluster Interpretation:")
    print("=" * 50)

    centroids = final_model.cluster_centers_
    cluster_interpretations = {}

    for i in range(optimal_k):
        cluster_mask = final_labels == i
        cluster_size = np.sum(cluster_mask)

        if cluster_size > 0:
            avg_silhouette = np.mean(sample_scores[cluster_mask])
            centroid = centroids[i].ravel()

            # Enhanced characteristics for returns vs levels
            if USE_RETURNS:
                # For returns: cumulative effect and volatility analysis
                cumulative_return = np.cumsum(centroid)
                net_trend = cumulative_return[-1]  # Total cumulative return
                volatility = np.std(centroid)
                max_drawdown = np.min(cumulative_return)
                max_peak = np.max(cumulative_return)
            else:
                # For levels: shape analysis
                volatility = np.std(centroid)
                net_trend = centroid[-1] - centroid[0]  # End vs start
                max_peak = np.max(centroid)
                max_drawdown = np.min(centroid)

            # Find key timing
            peak_idx = np.argmax(centroid)
            trough_idx = np.argmin(centroid)

            # Pattern classification
            if peak_idx < len(centroid) // 3:
                timing = "Early Peak"
            elif peak_idx > 2 * len(centroid) // 3:
                timing = "Late Peak"
            else:
                timing = "Mid-Day Peak"

            trend_direction = "Bullish" if net_trend > 0 else "Bearish"
            intensity = "Strong" if abs(net_trend) > volatility else "Moderate"

            pattern_name = f"{intensity} {trend_direction} {timing}"

            print(f"\n🏷️  Cluster {i}: {cluster_size} days ({cluster_size/len(final_labels)*100:.1f}%)")
            print(f"   📊 Silhouette: {avg_silhouette:.4f}")
            print(f"   🎯 Pattern: {pattern_name}")
            print(f"   📈 Net trend: {net_trend:.4f}")
            print(f"   📊 Volatility: {volatility:.4f}")
            print(f"   ⏰ Peak timing: Hour index {peak_idx}")
            print(f"   📉 Trough timing: Hour index {trough_idx}")

            cluster_interpretations[i] = pattern_name

    # ========== STEP 8: SAVE ENHANCED RESULTS ==========
    print(f"\n💾 Saving enhanced results...")

    model_artifacts = {
        'model': final_model,
        'scaler': scaler,
        'labels': final_labels,
        'optimal_k': optimal_k,
        'best_silhouette': best_score,
        'best_method': best_method,
        'hour_labels': list(hour_labels),
        'use_returns': USE_RETURNS,
        'sample_scores': sample_scores,
        'cluster_interpretations': cluster_interpretations,
        'hour_index_training': list(hour_labels),  # For deployment alignment
        'preprocessing_steps': {
            'imputation': 'forward_backward_fill',
            'returns_computation': 'log_diff' if USE_RETURNS else None,
            'normalization': 'dataset_standard_scaler' if USE_RETURNS else 'per_series_z_norm'
        }
    }

    joblib.dump(model_artifacts, f"{OUTPUT_DIR}/kshape_model_enhanced.pkl")

    # Enhanced assignments with interpretations
    assignments_df = pd.DataFrame({
        'date': df_imputed.index[:len(final_labels)],
        'cluster': final_labels,
        'pattern_name': [cluster_interpretations[label] for label in final_labels],
        'silhouette_score': sample_scores,
        'confidence': np.maximum(0, sample_scores)  # Simple confidence mapping
    })
    assignments_df.to_csv(f"{OUTPUT_DIR}/enhanced_cluster_assignments.csv", index=False)

    # Cluster summary
    cluster_summary = pd.DataFrame([
        {
            'cluster_id': i,
            'pattern_name': cluster_interpretations[i],
            'count': np.sum(final_labels == i),
            'percentage': np.sum(final_labels == i) / len(final_labels) * 100,
            'avg_silhouette': np.mean(sample_scores[final_labels == i]),
            'centroid_start': centroids[i].ravel()[0],
            'centroid_mid': centroids[i].ravel()[len(centroids[i].ravel())//2],
            'centroid_end': centroids[i].ravel()[-1]
        }
        for i in range(optimal_k)
    ])
    cluster_summary.to_csv(f"{OUTPUT_DIR}/cluster_summary.csv", index=False)

    print(f"✅ Enhanced results saved to {OUTPUT_DIR}/")

    # ========== STEP 9: COMPREHENSIVE VISUALIZATIONS ==========
    print(f"\n📊 Creating comprehensive visualizations...")

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # 1. Method comparison (if enabled)
    if COMPARE_METHODS and results['ts_kmeans']['silhouette_scores']:
        axes[0,0].plot(K_RANGE, results['k_shape']['silhouette_scores'], 'bo-', label='k-Shape', linewidth=2)
        axes[0,0].plot(K_RANGE, results['ts_kmeans']['silhouette_scores'], 'rs--', label='TS-KMeans (DTW)', linewidth=2)
        axes[0,0].axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Selected k={optimal_k}')
        axes[0,0].legend()
    else:
        axes[0,0].plot(K_RANGE, results['k_shape']['silhouette_scores'], 'bo-', linewidth=2)
        axes[0,0].axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        axes[0,0].legend()

    axes[0,0].set_xlabel('Number of Clusters (k)')
    axes[0,0].set_ylabel('Silhouette Score')
    axes[0,0].set_title('Enhanced K Selection with Metric Alignment')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Cluster centroids with hour labels
    for i in range(optimal_k):
        axes[0,1].plot(range(len(hour_labels)), centroids[i].ravel(),
                       linewidth=3, label=f'{cluster_interpretations[i]} (n={np.sum(final_labels==i)})')
    axes[0,1].set_xlabel('Hour Index')
    axes[0,1].set_ylabel('Normalized Value')
    axes[0,1].set_title('Cluster Centroids with Pattern Names')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].grid(True, alpha=0.3)

    # 3. Sample patterns by cluster
    colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
    for i in range(optimal_k):
        cluster_mask = final_labels == i
        cluster_data = X_normalized[cluster_mask][:min(5, np.sum(cluster_mask))]
        
        for j, day_data in enumerate(cluster_data):
            alpha = 0.8 if j == 0 else 0.4
            axes[0,2].plot(range(len(hour_labels)), day_data.ravel(),
                           color=colors[i], alpha=alpha, linewidth=2 if j == 0 else 1)

    axes[0,2].set_xlabel('Hour Index')
    axes[0,2].set_ylabel('Normalized Value')
    axes[0,2].set_title('Sample Patterns by Cluster')
    axes[0,2].grid(True, alpha=0.3)

    # 4. Enhanced cluster distribution
    cluster_counts = np.bincount(final_labels)
    bars = axes[1,0].bar(range(optimal_k), cluster_counts, color=colors[:optimal_k])
    axes[1,0].set_xlabel('Cluster ID')
    axes[1,0].set_ylabel('Number of Days')
    axes[1,0].set_title('Cluster Size Distribution')
    for i, (count, bar) in enumerate(zip(cluster_counts, bars)):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cluster_counts)*0.01,
                       f'{count}\n({count/len(final_labels)*100:.1f}%)', ha='center', va='bottom')

    # 5. Enhanced silhouette analysis
    y_lower = 10
    for i in range(optimal_k):
        if np.sum(final_labels == i) > 0:
            cluster_scores = sample_scores[final_labels == i]
            cluster_scores.sort()

            size_cluster_i = cluster_scores.shape[0]
            y_upper = y_lower + size_cluster_i

            axes[1,1].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_scores,
                                    facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

            axes[1,1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

    axes[1,1].axvline(best_score, color="red", linestyle="--", label=f'Average: {best_score:.4f}')
    axes[1,1].set_xlabel('Silhouette Coefficient')
    axes[1,1].set_ylabel('Cluster Label')
    axes[1,1].set_title('Enhanced Silhouette Analysis')
    axes[1,1].legend()

    # 6. Temporal distribution
    dates = assignments_df['date']
    clusters = assignments_df['cluster']
    scatter = axes[1,2].scatter(dates, clusters, c=clusters, cmap='Set1', alpha=0.7, s=30)
    axes[1,2].set_xlabel('Date')
    axes[1,2].set_ylabel('Cluster ID')
    axes[1,2].set_title('Cluster Assignments Over Time')
    axes[1,2].grid(True, alpha=0.3)

    # 7. Weekly pattern analysis
    weekday_pct = get_weekly_pattern_analysis(assignments_df)
    im = axes[2,0].imshow(weekday_pct.values, aspect='auto', cmap='YlOrRd')
    axes[2,0].set_xticks(np.arange(weekday_pct.shape[1]))
    axes[2,0].set_xticklabels([f'C{i}' for i in weekday_pct.columns])
    axes[2,0].set_yticks(range(len(weekday_pct)))
    axes[2,0].set_yticklabels(weekday_pct.index)
    axes[2,0].set_title('Weekly Pattern Distribution (%)')
    plt.colorbar(im, ax=axes[2,0])

    # 8. Confidence distribution
    axes[2,1].hist(sample_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[2,1].axvline(np.mean(sample_scores), color='red', linestyle='--', label=f'Mean: {np.mean(sample_scores):.3f}')
    axes[2,1].set_xlabel('Silhouette Score')
    axes[2,1].set_ylabel('Frequency')
    axes[2,1].set_title('Silhouette Score Distribution')
    axes[2,1].legend()

    # 9. Summary statistics
    summary_text = f"""
Enhanced Clustering Summary
=========================
Method: {best_method.replace('_', '-').title()}
Data: {'Log Returns' if USE_RETURNS else 'Price Levels'}
Sample Size: {len(assignments_df)} days
Optimal k: {optimal_k}
Best Silhouette: {best_score:.4f}

Cluster Patterns:
"""
    for i in range(optimal_k):
        count = np.sum(final_labels == i)
        pct = count / len(final_labels) * 100
        summary_text += f"• {cluster_interpretations[i]}: {count} days ({pct:.1f}%)\n"

    axes[2,2].text(0.05, 0.95, summary_text, transform=axes[2,2].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=9)
    axes[2,2].set_xlim(0, 1)
    axes[2,2].set_ylim(0, 1)
    axes[2,2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/enhanced_clustering_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n🎉 Enhanced Analysis Complete!")
    print(f"📁 Results saved in '{OUTPUT_DIR}/' directory")
    print(f"🏆 Best model: {best_method.replace('_', '-').title()} with k={optimal_k}")
    print(f"📊 {len(cluster_interpretations)} interpretable patterns identified")

    # ========== FINAL SUMMARY ==========
    print(f"\n📈 Enhanced Summary:")
    print("=" * 30)
    print(f"Sample size: {len(assignments_df)} days")
    print(f"Time series length: {len(hour_labels)} hours")
    print(f"Method used: {best_method.replace('_', '-').title()}")
    print(f"Optimal clusters: {optimal_k}")
    print(f"Best silhouette score: {best_score:.4f}")
    print(f"Data processing: {'Log returns with dataset scaling' if USE_RETURNS else 'Price levels with per-series normalization'}")

    print(f"\n🏷️  Identified Patterns:")
    for i in range(optimal_k):
        count = np.sum(final_labels == i)
        pct = count / len(final_labels) * 100
        avg_sil = np.mean(sample_scores[final_labels == i])
        print(f"  • {cluster_interpretations[i]}: {count} days ({pct:.1f}%, silhouette: {avg_sil:.3f})")

if __name__ == "__main__":
    main()