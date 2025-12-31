# chatbot-with-intent-classification
This repository contains code and notebooks for building an intent-classification chatbot. It demonstrates preprocessing, text vectorization (Word2Vec and SBERT), training classification models, and running a simple interactive classifier.

Contents
- README.md — this file
- utils/ — helper scripts and notebooks (preprocessing, vectorization, small utilities)
- data/ — CSV datasets (not committed if large)
- pickle_files/ — trained model artifacts (not committed by default)
- intent_classification.ipynb — exploratory notebook
- intent_classification_with_balanced_data.ipynb — notebook using balanced data

Quickstart (Windows)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or .venv\Scripts\activate.bat for cmd.exe
```

2. Install dependencies (create `requirements.txt` first if not present):

```powershell
python -m pip install -r requirements.txt
# or install packages individually:
python -m pip install numpy pandas scikit-learn nltk sentence-transformers gensim
```

3. Download or prepare data and models:

- dataset : [IDE functionalities text dataset](https://www.kaggle.com/datasets/abdullahusmani86/intent-classification-for-ide-functionalities?resource=download)
- Put large datasets in `data/` (recommended: keep out of git). See `data/` for CSV filenames.
- Trained models are in `pickle_files/`; these are large — consider storing them outside the repo or using Git LFS.

4. Run the interactive tester:

```powershell
python utils\test.py
```

5. Execution Flow

- utils / data_preprocessing.ipynb
- utils / vectorize_text.ipynb
- utils / encode_result_col.ipynb
- utils / balancing_data.ipynb
- ./ intent_classification.ipynb OR intent_classification_with_balanced_data.ipynb
- utils / test.py

Notes on data and models
- This project keeps several CSV files in `data/` and pickled model files in `pickle_files/`.
- Best practice: do NOT commit raw data and large binary model files to the Git repo. Instead:
	- Add `data/` and `pickle_files/` to `.gitignore`.
	- Provide `data/sample/` with a very small sample CSV for demos, or a `data/README.md` explaining how to obtain or regenerate data.
	- Use an artifact server, cloud storage, or Git LFS for large models.

What to commit
- Commit:
	- Source code (`*.py`) — e.g., `utils/test.py`.
	- Notebooks (`*.ipynb`) if you want to share experiments.
	- `README.md`, `requirements.txt`, and small sample data or scripts that reproduce preprocessing and training.
- Do NOT commit:
	- Virtual environments (e.g., `.venv/`)
	- Large datasets  amd pickle models(`data/*.csv`,`pickle_files/*.pkl`) unless you intentionally track them with LFS.

Suggested `.gitignore` entries
```
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
data/
pickle_files/
*.csv
*.pkl
/.vscode/
```

Reproducing models
- Use the notebooks in `utils/` to run preprocessing and training steps. They show how datasets are cleaned, vectorized, and how classifiers are trained.

Useful commands
- Check the Python interpreter being used:
```
where python   # Windows
```
- Ensure pip installs into the active interpreter:
```
python -m pip install <package>
```

Help / Next steps
- I can: generate a `requirements.txt`, update `.gitignore`, or prepare a small `data/sample/` demo file — tell me which you'd like.

License
- Add a license file if you plan to publish (e.g., MIT). If unsure, add `LICENSE` later.