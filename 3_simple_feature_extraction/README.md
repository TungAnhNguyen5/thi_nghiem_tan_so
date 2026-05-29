# Simple Feature Extraction

This mini project computes three simple features from interleaved float32 IQ `.dat` files:

- signal power
- occupied bandwidth
- spectral entropy

## Example

Generate a realistic sample capture:

```bash
python generate_sample_iq.py
```

Run the extractor on one file or a whole folder:

```bash
python main.py drone_sample_iq.dat
python main.py .
```

The script writes one CSV per input file into `output/` using the same stem name.

## Output

- `drone_sample_iq.dat` -> `output/drone_sample_iq.csv`

## Notes

- IQ files are stored as interleaved float32 values: real, imag, real, imag, ...
- The sample generator creates short bursts with slight frequency drift and noise so the features look more realistic than a single pure tone.