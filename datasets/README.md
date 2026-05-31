# Datasets

Place multi-view `.mat` files in this directory as `<dataset_name>.mat`.

Example:

```
datasets/BBCsports.mat
datasets/ALOI.mat
datasets/Prokaryotic.mat
datasets/100leaves.mat
```

## Supported dataset names

`Scene15`, `Caltech101-20`, `Orl_mtv`, `HandWritten`, `ALOI`, `Reuters_dim10`,
`NoisyMNIST-30000`, `2view-caltech101-8677sample`, `MNIST-USPS`,
`AWA-7view-10158sample`, `caltech7`, `BDGP`, `BBCsports`, `3Sources`,
`YouTube_X`, `HandWritten_X`, `100leaves`, `Prokaryotic`, `yale_mtv`, `feature_matrix`

You can also pass any `.mat` path via `--dataset_path`.

**Note:** Dataset files are large and not included in this repository; download or copy them from your experiment environment.
