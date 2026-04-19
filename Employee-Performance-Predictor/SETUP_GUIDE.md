# ⚡ COMPLETE SETUP & RUN GUIDE
## Employee Performance Predictor

---

## 🖥️ WINDOWS SETUP

### Step 1: Install Python
1. Go to https://python.org/downloads
2. Download Python 3.9 or newer
3. During install → ✅ CHECK "Add Python to PATH"
4. Verify: open Command Prompt → type: `python --version`

### Step 2: Open Project Folder
```cmd
cd path\to\Employee-Performance-Predictor
```
Example:
```cmd
cd C:\Users\YourName\Downloads\Employee-Performance-Predictor
```

### Step 3: Create Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```
You'll see `(venv)` appear in your terminal — that means it's active.

### Step 4: Install Libraries
```cmd
pip install -r requirements.txt
```
Wait for all packages to install (~2-3 minutes).

### Step 5: Run the Project
```cmd
python main.py
```

---

## 🍎 MAC / LINUX SETUP

### Step 1: Install Python (Mac)
```bash
brew install python3
# OR download from python.org
```

### Step 2: Open Terminal in Project Folder
```bash
cd ~/Downloads/Employee-Performance-Predictor
```

### Step 3: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Libraries
```bash
pip install -r requirements.txt
```

### Step 5: Run the Project
```bash
python main.py
```

---

## 🎯 WHAT HAPPENS WHEN YOU RUN

```
[STEP 1/5] Generating Synthetic Employee Dataset...
  → Creates data/employee_data.csv (1000 employees, 20 features)

[STEP 2/5] Running Exploratory Data Analysis...
  → Saves 9 charts to images/ folder

[STEP 3/5] Training Machine Learning Models...
  → Trains 5 ML models: LR, DT, RF, GB, SVM
  → Selects best model
  → Saves to models/best_model.pkl

[STEP 4/5] Running Predictions...
  → Predicts single employee performance
  → Runs batch prediction on 100 employees
  → Saves results to outputs/predictions.csv

[STEP 5/5] Summary
  → Shows final accuracy and output files
```

---

## 📁 OUTPUT FILES AFTER RUNNING

| File | Location | Description |
|------|----------|-------------|
| employee_data.csv | data/ | Raw 1000-employee dataset |
| predictions.csv | outputs/ | Predicted performance for 100 employees |
| model_comparison.csv | outputs/ | Accuracy of all 5 models |
| best_model.pkl | models/ | Saved trained model |
| *.png (9 files) | images/ | All visualization charts |

---

## 🔧 RUNNING INDIVIDUAL STEPS

```bash
# Only generate EDA charts
python main.py --eda

# Only train models
python main.py --train

# Only run predictions (requires trained model)
python main.py --predict
```

---

## 🐛 COMMON ERRORS & FIXES

### Error: "ModuleNotFoundError: No module named 'pandas'"
**Fix:** Run `pip install -r requirements.txt`

### Error: "python is not recognized"
**Fix:** During Python install, make sure "Add to PATH" was checked.
Or use `python3` instead of `python`.

### Error: "No such file or directory: data/employee_data.csv"
**Fix:** Run `python main.py` (full pipeline) first, not `--predict` alone.

### Error: "Permission denied"
**Fix (Windows):** Run Command Prompt as Administrator.
**Fix (Mac/Linux):** Use `sudo pip install -r requirements.txt`

### Error: Port already in use (if running Jupyter)
**Fix:** `jupyter notebook --port=8889`

---

## 📓 RUNNING JUPYTER NOTEBOOK (Optional)

```bash
# Install Jupyter
pip install jupyter

# Launch
jupyter notebook

# Open: notebooks/analysis.ipynb
```

---

## 🚀 GITHUB UPLOAD (AFTER PROJECT RUNS)

```bash
# Initialize git
git init
git add .
git commit -m "Initial commit: Employee Performance Predictor - complete ML pipeline"

# Connect to your GitHub repo
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Employee-Performance-Predictor.git
git push -u origin main
```

---

## ✅ VERIFICATION CHECKLIST

After running `python main.py`, verify:

- [ ] `data/employee_data.csv` exists (1000 rows)
- [ ] `images/` has 9 PNG files
- [ ] `models/best_model.pkl` exists
- [ ] `outputs/predictions.csv` exists (100 rows + PredictedPerformance column)
- [ ] Terminal shows accuracy > 70%
- [ ] No red error messages

If all boxes are checked → your project is working perfectly! ✅
