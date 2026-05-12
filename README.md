# Health App (HealthCareML)

Short summary
---------------

This repository contains a lightweight Flask backend that predicts probable diseases from user-entered symptoms and returns supporting information (description, precautions, medications, diet and workout recommendations). The backend serves a small frontend (templates + static assets) and uses a scikit-learn model stored at `Backend/model/svc.pkl` (a dummy model is included by default).

Key features
------------
- Enter comma-separated symptoms on the web form and receive a suggested diagnosis.
- Helper data (datasets) provide brief description, precautions, medications and diet/workout suggestions for the predicted disease.
- Small, self-contained Flask app in `Backend/` for local testing and development.

Folder layout
-------------

- `Backend/` – main Flask application, dataset CSVs, model generator, helper scripts, and `run_backend.ps1` to start the app.
- `archive/` – (created by cleanup) contains backed-up copies of previously duplicated top-level folders and notebooks.

Prerequisites
-------------

- Python 3.9+ (3.11/3.13 tested here)
- PowerShell (Windows) for the included helper scripts
- pip installed packages listed in `Backend/requirements.txt`

Run (development)
-----------------

Recommended: run inside a Python virtual environment. The helper PowerShell script `Backend/run_backend.ps1` automates install and launch. Example (PowerShell):

```powershell
# From repository root
cd "C:\Users\Souvi\Desktop\health_app\Backend"
# Install deps
python -m pip install -r requirements.txt
# Generate dummy model (if missing)
python generate_model_from_module.py
# Run the Flask app
python main.py
# Or use the helper script which runs the three steps above:
.\\run_backend.ps1
```

Alternative (single helper command - PowerShell):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_backend.ps1
```

Open the app
-------------

Visit http://127.0.0.1:5000 in your browser. The form accepts name, age, location and a free-text `symptoms` field (comma-separated). Submitting the form returns the predicted disease and supporting info.

Notes & troubleshooting
-----------------------
- The repository includes a dummy model generator that creates `Backend/model/svc.pkl` for testing. Replace it with your trained `svc.pkl` to get real predictions.
- If TextBlob reports missing corpora, run:

```powershell
python -m textblob.download_corpora
```

- The app uses the Flask development server (not for production). For production use a WSGI server such as Gunicorn or Waitress.
- The `archive/` folder was created during cleanup and contains older copies of top-level `Frontend/`, `templates/`, `static/`, `model/`, and notebooks. No data was permanently deleted.

What I changed
---------------
- Added this top-level README with a short summary and run instructions.

If you'd like, I can:
- Create a `.gitignore` and commit these changes.
- Update `Backend/README.md` with the same run steps.
- Create a single PowerShell script that starts the app and opens the browser automatically.
