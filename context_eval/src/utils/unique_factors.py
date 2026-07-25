import pandas as pd
import glob
import os

DATA_DIR = 'survey_results'
files = sorted(
    glob.glob(os.path.join(DATA_DIR, 'likelihood_survey_*.xlsx')) +
    glob.glob(os.path.join(DATA_DIR, 'likelihood_surveys_*.xlsx')) +
    glob.glob(os.path.join(DATA_DIR, 'linkelihood_survey_*.xlsx'))
)

whiches = ["First", "Second", "Third", "Fourth", "Fifth"]
factor_set = set()

for file in files:
    xl = pd.ExcelFile(file)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        for col in df.columns:
            if any(w in col for w in whiches):
                for v in df[col].dropna():
                    s = str(v).strip()
                    if s and s.lower() != "nan":
                        factor_set.add(s)

print("Unique factor names found in your survey files:")
for f in sorted(factor_set):
    print(f"- {f}")
print(f"\nTotal unique: {len(factor_set)}")
