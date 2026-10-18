# NeuroVision AI: brain tumor classification from MRI

NeuroVision AI classifies brain MRI images into four classes: **glioma**, **meningioma**, **pituitary tumor** and **no tumor**. It is a decision-support tool for doctors and never replaces them. The model is a CNN built with TensorFlow/Keras, OpenCV handles image preprocessing, and a Streamlit app lets you test predictions.

## Results

| Metric | Value |
|---|---|
| Test accuracy | TODO |
| Macro F1-score | TODO |
| Parameters | 257,860 |
| Training time | TODO |
| Image size | TODO (224x224 or 128x128) |

Figures are in `reports/`: class distribution, samples, learning curves, confusion matrix, correct and incorrect predictions, Grad-CAM.

## Project structure

```
NeuroVision/
├── app/
│   └── streamlit_app.py        Streamlit interface
├── data/
│   └── raw/                    glioma/ meningioma/ notumor/ pituitary/  (not versioned)
├── models/
│   └── best_model.keras        trained model
├── notebooks/
│   ├── EDA.ipynb               data exploration and cleaning
│   ├── preprocessing.ipynb     loading, split, normalization, augmentation preview
│   ├── model_training.ipynb    model, training, learning curves
│   └── 04_evaluation.ipynb     test evaluation, error analysis, Grad-CAM, predict_image
├── reports/                    figures
├── src/neurovision/
│   ├── data.py                 scan, validation, duplicates, loading
│   ├── preprocessing.py        label encoding, tf.data pipeline, train/val/test split
│   ├── model.py                CNN definition
│   ├── predict.py              predict_image and helpers
│   └── gradcam.py              explainability
├── pyproject.toml
├── uv.lock
└── README.md
```

## Dataset

Four folders, one per class. The folder name is the class label. The dataset is not stored in this repository. **TODO: add the source and link.**

Place it like this before running anything:

```
data/raw/glioma/
data/raw/meningioma/
data/raw/notumor/
data/raw/pituitary/
```

## Installation

The project uses [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/ABDELHAFIDaz/NeuroVision.git
cd NeuroVision
uv sync
```

## How to run

1. **Explore and clean the data:** run `notebooks/EDA.ipynb`.
2. **Check preprocessing:** run `notebooks/preprocessing.ipynb`.
3. **Train:** run `notebooks/model_training.ipynb`. The best model is saved to `models/best_model.keras`. If training is interrupted, run the training cell again and it resumes from the last finished epoch.
4. **Evaluate:** run `notebooks/04_evaluation.ipynb`. The test set is used once, here.
5. **Launch the app:**

```
uv run streamlit run app/streamlit_app.py
```

Upload an MRI image (jpg, jpeg, png or bmp) to get the predicted class, the confidence and the probability for each class. The Grad-CAM option shows which regions influenced the prediction.

## Predict from code

```python
from neurovision.predict import predict_image

predict_image("path/to/image.jpg")
```

Invalid inputs raise a clear error: `FileNotFoundError` for a missing file and `ValueError` for an unsupported or unreadable image.

## Methodology

- **Data cleaning:** extension check (jpeg, jpg, bmp, png), corrupted files handled with try/except, byte-identical duplicates removed (194 groups found, 7023 to 6726 images).
- **Class balance:** after cleaning, the largest class is about 1.07 times the smallest. `class_weight` is computed on the training set only.
- **Split:** stratified 70/15/15 (train, validation, test), fixed seed, done before augmentation to prevent data leakage.
- **Preprocessing:** resize with OpenCV, BGR to RGB, normalization to [0, 1] inside the model (`Rescaling`).
- **Augmentation (training only):** small rotation (about 11 degrees), zoom 10%, shift 5%.
- **Architecture:** 4 blocks of Conv2D + MaxPooling2D (32, 64, 128, 128 filters), GlobalAveragePooling2D, Dense(128), Dropout(0.5), Dense(4, softmax).
- **Training:** Adam (learning rate 1e-3), categorical cross-entropy, `ModelCheckpoint`, `EarlyStopping`, `BackupAndRestore`.
- **Pipeline:** `tf.data` with cache, shuffle (train only), batch and prefetch.

## Challenges

- Connecting VS Code to Colab worked, but sending the dataset failed (`drive.mount` error, zip upload failure, unresponsive kernel). Training was done **TODO: locally on CPU / on Kaggle**.
- 194 duplicate groups changed the apparent class balance, so duplicate removal comes before any split.

## Limitations

- Only exact duplicates are detected. Near-duplicates would need a perceptual hash.
- This tool is for decision support only and is not a certified medical device.