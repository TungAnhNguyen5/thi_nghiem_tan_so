# Unsupervised Clustering

This demo turns IQ recordings into unlabeled feature windows and groups similar windows with clustering algorithms.

## What it does

- loads interleaved float32 IQ `.dat` files
- splits each capture into fixed-size overlapping windows
- extracts time-domain and frequency-domain features per window
- clusters the window features with one of:
  - k-means
  - Gaussian mixture model
  - DBSCAN
  - optional HDBSCAN if installed
- saves a CSV with per-window features and cluster labels
- saves a PCA scatter plot for quick inspection

## Example

Run on a directory of IQ files:

```bash
python main.py ../3_simple_feature_extraction/output
```

Run on a single IQ capture:

```bash
python main.py ../3_simple_feature_extraction/output/drone_sample_iq.dat
```

## Notes

- K-means and GMM can optionally auto-select the number of clusters by silhouette score.
- DBSCAN and HDBSCAN are useful when you do not know the cluster count in advance.
- HDBSCAN is optional; install it separately if you want to use that algorithm.