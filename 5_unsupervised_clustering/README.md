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

Tune windowing and algorithm:

```bash
python main.py ../3_simple_feature_extraction/output/drone_sample_iq.dat \
  --fs 1000000 \
  --window-size 4096 \
  --hop-size 2048 \
  --algorithm kmeans \
  --cluster-count auto

python main.py ../3_simple_feature_extraction/output/drone_sample_iq.dat \
  --algorithm dbscan \
  --dbscan-eps 0.8 \
  --min-samples 6
```

## Code layout

The implementation is split into small modules to keep `main.py` short:

- `windowing.py` — window slicing (`window_iq`)
- `features.py` — feature extraction (`extract_features`, `build_feature_matrix`, `FEATURE_NAMES`)
- `clustering.py` — scaling/selection/clustering (`scale_features`, `choose_cluster_count`, `fit_clusterer`)
- `viz.py` — PCA projection + scatter plot (`project_for_plot`, `plot_clusters`)
- `io_utils.py` — IQ loading + CSV output (`load_iq`, `write_window_csv`)

## Notes

- K-means and GMM can optionally auto-select the number of clusters by silhouette score.
- DBSCAN and HDBSCAN are useful when you do not know the cluster count in advance.
- HDBSCAN is optional; install it separately if you want to use that algorithm.