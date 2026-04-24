# Medical Imaging CAD — Multi-Modality Diagnostic Aid

A Django-based computer-aided diagnosis (CAD) web application that runs deep learning models on medical imagery and returns probability-scored predictions. Built as the final-year B.E. Major Project at Lords Institute of Engineering & Technology (Osmania University), 2023–24.

> **Academic title:** *Research on the Application of Artificial Intelligence in Medical Imaging Diagnosis*
> **Team:** 3 members — Mohammed Faraz Uddin Fiyaz, Syed Khundmir Saquib, Mohammed Muzaffar Lateef
> **Guide:** Ms. Shahawar Fatima (Asst. Prof.), HOD: Dr. Abdul Rasool MD

---

## What it does

Uploads a medical image through a Django web frontend and routes it to the correct trained model based on modality. Four clinical pipelines:

| Modality | Task | Model | Dataset |
|---|---|---|---|
| Chest X-ray | 14-class multi-label pathology classification | DenseNet121 (ImageNet-pretrained, fine-tuned) | NIH ChestX-ray14 |
| Brain MRI | Binary tumor / no-tumor classification | CNN | Public brain-tumor MRI dataset |
| Mammography | Binary benign / malignant classification | CNN | Public mammography dataset |
| Stroke MRI | Stroke detection on brain MRI | CNN | Public stroke MRI dataset |

The chest X-ray pipeline is the scientific core of the project — it handles 14 pathologies simultaneously (cardiomegaly, emphysema, effusion, hernia, infiltration, mass, nodule, atelectasis, pneumothorax, pleural thickening, pneumonia, fibrosis, edema, consolidation) using a single multi-label classifier rather than 14 separate binary models.

## Technical highlights

- **Transfer learning**: DenseNet121 backbone pretrained on ImageNet, topped with Global Average Pooling and a 14-unit sigmoid dense layer for multi-label output
- **Class imbalance handling**: custom weighted binary cross-entropy loss (`get_weighted_loss`) computes per-class positive and negative frequencies from the training set and weights the loss inversely — necessary because NIH ChestX-ray14 is severely imbalanced (e.g., Hernia ~0.2% prevalence, Infiltration ~17%)
- **Leakage prevention**: patient-level train/test split check (`check_for_leakage`) ensures no patient ID appears in both splits — critical because NIH ChestX-ray14 contains multiple images per patient
- **Evaluation**: ROC curves and AUC per pathology via scikit-learn
- **Interpretability scaffolding**: Grad-CAM visualization code present (commented) to highlight which image regions drove the prediction
- **Deployment**: Django 4.1.7 + SQLite wrapping the four inference pipelines behind authenticated user/admin views

## Tech stack

Python 3.10 · TensorFlow 2.11 · Keras 2.11 · scikit-learn · NumPy · pandas · OpenCV · Pillow · Matplotlib · Seaborn · Django 4.1.7 · SQLite

## Repository layout

```
medical-imaging-cad/
├── medicals/                    # Django project root
│   ├── manage.py
│   ├── medicals/                # Django settings, URLs, WSGI
│   ├── users/                   # Patient-facing views
│   │   └── utility/
│   │       ├── predictChest.py        # Chest X-ray inference
│   │       ├── predictMammography.py  # Mammography inference
│   │       └── predictMriStroke.py    # Brain MRI / stroke inference
│   ├── admins/                  # Admin views
│   └── DP Model test.ipynb      # Training notebook — DenseNet121 on NIH ChestX-ray14
└── README.md
```

## What's NOT in the repo (and why)

Model weights (`.h5` files) and raw imaging datasets are **intentionally excluded** via `.gitignore`:

- `ChestModel.h5`, `DenseNet.h5`, `DenseNet-BC.h5`, `pretrainedmodel.h5`, `brain_tumor_detector.h5`, `mammography_model3.h5` — binary weight files not suitable for version control
- NIH ChestX-ray14 images (~45 GB, 112k frontal chest X-rays across 30k patients)
- Django media/ upload directory

To reproduce:
1. Download NIH ChestX-ray14 from https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Retrain DenseNet121 using the training notebook (`DP Model test.ipynb`)
3. Place resulting `.h5` files into `medicals/media/` per the paths referenced in `predictChest.py`, `predictMriStroke.py`, `predictMammography.py`

## Running the Django app (after weights are restored)

```bash
cd medicals
python -m venv venv && source venv/bin/activate        # Linux/Mac
# or:  venv\Scripts\activate                            # Windows
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Then open `http://127.0.0.1:8000/`.

## Status & limitations

This was a **3-person academic team project** submitted for a Bachelor's degree, not production-grade clinical software. It demonstrates an end-to-end pipeline from dataset curation → transfer learning → evaluation → web deployment, but **is not validated for real-world diagnostic use** and must not be used to inform medical decisions.

Academic baseline: project delivered against IEEE conference paper (July 2022, ISBN 978-1-6654-8192-2) as its scientific reference.

## License

MIT — see [LICENSE](LICENSE).
