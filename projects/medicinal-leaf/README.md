# Medicinal Leaf Classification — reconstructed baseline

Extract RGB/HSV histograms, horizontal symmetric GLCM statistics, eight-neighbor LBP, and Hu moments from a green foreground mask. Train a scaled PCA/random-forest pipeline. Scaling and PCA fit only the training split.

```sh
python3 -m pip install -r requirements.txt
python3 train.py --demo
python3 train.py --data /path/to/leaf_images
```

Place images in `leaf_images/class_name/*.jpg` (PNG/BMP also supported). Use enough examples per class for stratified splitting. The synthetic demo generates three simple shape/color classes; its score does not measure botanical recognition. Output model and metrics go to `artifacts/`.

This implementation uses 64×64 feature images and a heuristic mask. It does not recover the original 80-class experiment, augmentation, SMOTE, CNN, or t-SNE/UMAP analysis. The original notebook is already available in the Feature-Engineering-Project repository linked in the portfolio. This folder supplies a smaller runnable reconstruction.
