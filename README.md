#  Brain Stroke Classification using MRI Images

This project applies a **Convolutional Neural Network (CNN)** to classify brain MRI scans into three categories:
-  **Haemorrhagic Stroke**
-  **Ischemic Stroke**
-  **Normal Brain**

---

##  Dataset

The dataset is from [Kaggle - Brain Stroke MRI Images](https://www.kaggle.com/datasets/mitangshu11/brain-stroke-mri-images).

After downloading and extracting the dataset, place it inside the following local path:

```
data/Dataset_MRI_Folder/
```

Folder structure should look like:

```
Dataset_MRI_Folder/
 ├── Haemorrhagic/
 ├── Ischemic/
 └── Normal/
```

>  Note: The dataset is **not uploaded** to GitHub because of its large size.  
> The `data/` folder is included in `.gitignore` to avoid accidental upload.

---

##  Model Overview

A simple CNN model was built using **TensorFlow / Keras**, consisting of:
- Two convolutional + max-pooling layers  
- One dense layer (128 units) + Dropout(0.5)  
- Output layer with 3-class softmax activation

**Optimizer:** Adam (learning rate = 0.001)  
**Loss function:** categorical_crossentropy

---

##  Setup & Installation

Clone the repository:
```bash
git clone https://github.com/zahra-ghasemlou/stroke-mri-classification.git
cd stroke-mri-classification
```

Install dependencies:
```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install manually:
```bash
pip install tensorflow matplotlib numpy pillow
```

---

##  Run Training

Make sure the dataset folder is placed as described above (inside `data/Dataset_MRI_Folder/`), then run:

```bash
python src/project1.py
```

By default the script will:
- Load images using `ImageDataGenerator` with `validation_split=0.2`  
- Train the CNN for 10 epochs  
- Plot training & validation accuracy graphs

If your script file has a different name or is in a different path, run the correct filename.

---

##  Results

An example result image `example_result.jpg` is included to show expected plots.  
Final accuracy will vary depending on preprocessing, train/validation split and hyperparameters.  
(Example: ~76% validation accuracy after 10 epochs — your results may differ.)

---

##  Recommended Files in This Repo

- `src/project1.py` — main training & data-loading script  
- `requirements.txt` — Python dependencies  
- `example_result.jpg` — sample plot image (optional)  
- `.gitignore` — to exclude `data/` and other unneeded files

Sample `.gitignore`:
```
data/
__pycache__/
*.pyc
*.h5
```

---

##  Technologies

- Python 3.9+  
- TensorFlow / Keras  
- NumPy, Matplotlib, Pillow

---

##  Author

**Zahra Ghasemlou**  
For research and educational purposes.  
GitHub: [github.com/zahra-ghasemlou](https://github.com/zahra-ghasemlou)

---

