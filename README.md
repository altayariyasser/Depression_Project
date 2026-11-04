# Student Lifestyle and Depression Classification

An applied machine-learning project exploring student lifestyle data and comparing classification approaches. The repository includes the original notebook, dataset, and interactive Streamlit application.

## Start here

| Resource | Purpose |
|---|---|
| [Research notebook](Depression_Project.ipynb) | Exploratory analysis, cleaning, preprocessing, PCA, model comparison, and evaluation |
| [Interactive application](app.py) | Streamlit application source |
| [Dataset](student_lifestyle_100k.csv) | Existing project data |
| [Dependencies](requirements.txt) | Application dependencies |

## Workflow

The notebook explores lifestyle variables and class balance, prepares numeric and categorical features, and compares logistic regression and random forests with and without PCA. It examines precision, recall, F1, accuracy, and confusion matrices. Recorded results and their context remain in the original notebook.

## Run the application

```bash
git clone https://github.com/altayariyasser/Depression_Project.git
cd Depression_Project
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The notebook contains an original machine-specific CSV path; point it to the included dataset when running it elsewhere. Application dependencies are listed separately from any additional notebook visualization packages. This organization update does not rerun or validate the models or application.

## Project scope

This is an educational data-analysis and classification project. The notebook preserves its original observations and limitations; no new model-performance claims are introduced here.

[View the portfolio project directory](https://github.com/altayariyasser)
