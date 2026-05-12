# Health Care Center (Backend)

This folder contains a Flask backend that serves templates and predicts diseases from symptoms.

Quick start (PowerShell):

```powershell
cd "C:\Users\Souvi\Desktop\project git\Backend"
# Install dependencies
python -m pip install -r requirements.txt
# Create a dummy model (svc.pkl) used for testing
python generate_model_from_module.py
# Run the app
python main.Roshni.py
```

The server runs by default on http://127.0.0.1:5000.

Notes:
- The repository now contains a fake/dummy model at `model/svc.pkl` that always predicts "Fungal infection". Replace it with the real model to get actual predictions.
- Dataset CSVs are located in `dataset/`. If you have richer CSVs in `Backend/Dataset/`, they were copied into `Backend/dataset/` during setup.
- If TextBlob reports missing corpora, run:

```powershell
python -m textblob.download_corpora
```

If you want me to (choose one):
- Replace the dummy model with a `svc.pkl` you provide.
- Run end-to-end tests of the `/predict` endpoint with sample inputs and show results.
- Convert the frontend into Flask templates fully (I already copied simplified templates).