# `thi_nghiem_tan_so`

Small signal-processing practice workspace with four mini projects:

- `1_fft_visualizer/` - generates a noisy sine-wave FFT plot
- `2_uav_spectrogram/` - renders spectrograms from interleaved float32 IQ `.dat` files
- `3_simple_feature_extraction/` - extracts signal power, bandwidth, and entropy from IQ `.dat` files
- `4_pca_visualization/` - applies PCA to synthetic UAV RF feature vectors and plots class clusters
- `5_unsupervised_clustering/` - windows IQ captures, extracts features, and clusters them without labels

## Requirements

Use the repo virtual environment:

```bash
source /workspaces/codespaces-blank/thi_nghiem_tan_so/.venv/bin/activate
```

## Run the projects

FFT visualizer:

```bash
cd 1_fft_visualizer
python fft_visualizer.py
```

UAV spectrogram:

```bash
cd 2_uav_spectrogram
python main.py
```

Feature extraction:

```bash
cd 3_simple_feature_extraction
python main.py ../2_uav_spectrogram
```

PCA visualization:

```bash
cd 4_pca_visualization
python pca_visualizer.py
```

Unsupervised clustering:

```bash
cd 5_unsupervised_clustering
python main.py ../3_simple_feature_extraction/output
```

## Output

- `1_fft_visualizer/fft_visualizer.png`
- `2_uav_spectrogram/output/*.png`
- `3_simple_feature_extraction/output/*.csv`
- `4_pca_visualization/pca_clusters.png`
- `5_unsupervised_clustering/output/*.csv`
- `5_unsupervised_clustering/output/*.png`

## Notes

- IQ files are stored as interleaved float32 values: real, imag, real, imag, ...
- `3_simple_feature_extraction` writes one CSV per input file using the same stem name.