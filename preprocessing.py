"""
preprocessing.py
----------------
Data loading (load_dataset) and data cleaning (preprocess_data).

Data-quality problems are never hidden: every cleaning step is counted and
returned in a summary dictionary so the user can see what happened.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_generator import COLUMNS, DATASET_PATH, save_dataset

SALARY_COLUMNS = ["Salary_Min_LPA", "Salary_Max_LPA", "Salary_Midpoint_LPA"]
NOT_SPECIFIED = "Not Specified"

# Canonical spellings so "full-time" and "Full-time" are not counted separately.
EMPLOYMENT_TYPE_MAP = {
    "full-time": "Full-time", "full time": "Full-time", "fulltime": "Full-time",
    "part-time": "Part-time", "part time": "Part-time", "parttime": "Part-time",
    "internship": "Internship", "intern": "Internship",
    "contract": "Contract", "contractual": "Contract",
}
WORK_MODE_MAP = {
    "remote": "Remote", "work from home": "Remote", "wfh": "Remote",
    "hybrid": "Hybrid", "on-site": "On-site", "onsite": "On-site",
    "on site": "On-site", "office": "On-site",
}
YES_VALUES = {"yes", "y", "true", "1"}
NO_VALUES = {"no", "n", "false", "0"}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_dataset(path=None, auto_generate=True):
    """
    Load the job-market CSV with pandas.

    - If the file is missing and auto_generate is True, the SYNTHETIC dataset is
      generated locally and saved first.
    - Returns a DataFrame, or None (after printing an error message) on failure.
    """
    path = Path(path) if path else DATASET_PATH

    if not path.exists():
        if not auto_generate:
            print(f"ERROR: Dataset file not found: {path}")
            return None
        print(f"Dataset file not found: {path}")
        print("Generating the SYNTHETIC sample dataset locally (not real job data)...")
        try:
            save_dataset(path)
        except OSError as error:
            print(f"ERROR: Could not create the dataset file: {error}")
            return None

    try:
        # Read everything as text first; preprocess_data converts types safely.
        data = pd.read_csv(path, dtype=str, encoding="utf-8")
    except pd.errors.EmptyDataError:
        print(f"ERROR: The file is empty: {path}")
        return None
    except pd.errors.ParserError as error:
        print(f"ERROR: The CSV file is malformed and could not be parsed: {error}")
        return None
    except UnicodeDecodeError:
        print("ERROR: The file is not valid UTF-8 text. Please re-save it as CSV (UTF-8).")
        return None
    except (OSError, ValueError) as error:
        print(f"ERROR: Could not read the dataset: {error}")
        return None

    missing_columns = [c for c in COLUMNS if c not in data.columns]
    if missing_columns:
        print(f"ERROR: Invalid dataset. Missing columns: {', '.join(missing_columns)}")
        return None

    print(f"Dataset loaded successfully: {len(data)} records, {len(data.columns)} attributes.")
    print("NOTE: This is SYNTHETIC sample data, not real-time job-market data.")
    return data


# --------------------------------------------------------------------------
# Data cleaning
# --------------------------------------------------------------------------
def _clean_text_series(series):
    """Trim and collapse whitespace; turn empty strings into NaN. Returns (series, n_trimmed)."""
    original = series.astype("object")
    cleaned = original.map(
        lambda v: " ".join(v.split()) if isinstance(v, str) else v
    )
    cleaned = cleaned.map(lambda v: np.nan if isinstance(v, str) and v == "" else v)
    is_text = original.map(lambda v: isinstance(v, str)).astype(bool)
    changed = is_text & cleaned.notna() & (original != cleaned)
    return cleaned, int(changed.sum())


def _normalise_skills(value):
    """'Python ,SQL,  Git' -> 'Python, SQL, Git' (ignores empty items)."""
    if not isinstance(value, str):
        return value
    items = [item.strip() for item in value.split(",") if item.strip()]
    return ", ".join(items) if items else np.nan


def preprocess_data(data):
    """
    Clean the raw dataset.

    Returns (cleaned_dataframe, summary_dict). The summary lists every
    data-quality problem found and what was done about it.
    """
    df = data.copy()
    summary = {"records_before": len(df)}
    issues = []  # human-readable warnings

    # 1. Trim spaces / convert empty text to missing -------------------------
    text_columns = [c for c in COLUMNS if c not in SALARY_COLUMNS + ["Posted_Date"]]
    trimmed_cells = 0
    for column in text_columns:
        df[column], changed = _clean_text_series(df[column])
        trimmed_cells += changed
    summary["cells_with_extra_spaces_fixed"] = trimmed_cells
    if trimmed_cells:
        issues.append(f"Removed extra spaces from {trimmed_cells} text cells.")

    # Record missing values per column (before filling anything).
    missing_counts = {c: int(df[c].isna().sum()) for c in COLUMNS if df[c].isna().sum() > 0}
    summary["missing_values_found"] = missing_counts
    if missing_counts:
        details = ", ".join(f"{c}: {n}" for c, n in missing_counts.items())
        issues.append(f"Missing/empty values found -> {details}.")

    # 2. Records without Job_ID cannot be identified -> removed (and reported)
    no_id = int(df["Job_ID"].isna().sum())
    df = df[df["Job_ID"].notna()]
    summary["removed_missing_job_id"] = no_id
    if no_id:
        issues.append(f"Removed {no_id} records with no Job_ID.")

    # 3. Duplicate Job_ID values (first occurrence is kept)
    duplicates = int(df.duplicated(subset="Job_ID", keep="first").sum())
    df = df.drop_duplicates(subset="Job_ID", keep="first")
    summary["duplicate_job_ids_removed"] = duplicates
    if duplicates:
        issues.append(f"Removed {duplicates} duplicate Job_ID records.")

    # 4. Job_Title is essential for every analysis -> remove records without it
    no_title = int(df["Job_Title"].isna().sum())
    df = df[df["Job_Title"].notna()]
    summary["removed_missing_job_title"] = no_title
    if no_title:
        issues.append(f"Removed {no_title} records with no Job_Title.")

    df = df.copy()

    # 5. Standardise categorical text and fill remaining gaps ---------------
    df["Skills"] = df["Skills"].map(_normalise_skills)
    df["Employment_Type"] = df["Employment_Type"].map(
        lambda v: EMPLOYMENT_TYPE_MAP.get(v.lower(), v) if isinstance(v, str) else v)
    df["Work_Mode"] = df["Work_Mode"].map(
        lambda v: WORK_MODE_MAP.get(v.lower(), v) if isinstance(v, str) else v)

    def _normalise_yes_no(value):
        if not isinstance(value, str):
            return NOT_SPECIFIED
        if value.lower() in YES_VALUES:
            return "Yes"
        if value.lower() in NO_VALUES:
            return "No"
        return NOT_SPECIFIED

    df["Fresher_Friendly"] = df["Fresher_Friendly"].map(_normalise_yes_no)
    fill_columns = ["Company", "Location", "Skills", "Experience",
                    "Employment_Type", "Work_Mode", "Education"]
    summary["missing_text_filled_with_not_specified"] = int(df[fill_columns].isna().sum().sum())
    df[fill_columns] = df[fill_columns].fillna(NOT_SPECIFIED)

    # 6. Salary columns -> numeric ------------------------------------------
    non_numeric_total = 0
    for column in SALARY_COLUMNS:
        raw = df[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        # Text that is present but cannot be read as a number (e.g. "abc")
        non_numeric_total += int((numeric.isna() & raw.notna()).sum())
        df[column] = numeric
    summary["non_numeric_salary_values"] = non_numeric_total
    if non_numeric_total:
        issues.append(f"{non_numeric_total} salary values were not numeric and are treated as missing.")

    # 6a. Negative salaries are invalid -> set to missing (record is kept)
    negative_mask = (df[SALARY_COLUMNS] < 0).any(axis=1)
    summary["records_with_negative_salary"] = int(negative_mask.sum())
    for column in SALARY_COLUMNS:
        df.loc[df[column] < 0, column] = np.nan
    if negative_mask.any():
        issues.append(f"{int(negative_mask.sum())} records had NEGATIVE salaries; "
                      "those salary values are ignored in salary statistics.")

    # 6b. Invalid range: minimum greater than maximum
    range_mask = df["Salary_Min_LPA"] > df["Salary_Max_LPA"]
    summary["records_with_invalid_salary_range"] = int(range_mask.sum())
    df.loc[range_mask, SALARY_COLUMNS] = np.nan
    if range_mask.any():
        issues.append(f"{int(range_mask.sum())} records had Salary_Min greater than Salary_Max; "
                      "their salary values are ignored in salary statistics.")

    # 6c. Rebuild a missing midpoint when min and max are valid
    can_rebuild = (df["Salary_Midpoint_LPA"].isna()
                   & df["Salary_Min_LPA"].notna() & df["Salary_Max_LPA"].notna())
    df.loc[can_rebuild, "Salary_Midpoint_LPA"] = (
        (df["Salary_Min_LPA"] + df["Salary_Max_LPA"]) / 2).round(2)
    summary["midpoints_recalculated"] = int(can_rebuild.sum())

    # Records that still have no usable salary stay in the data but are skipped in salary maths
    no_salary = int(df[SALARY_COLUMNS].isna().any(axis=1).sum())
    summary["records_without_valid_salary"] = no_salary
    if no_salary:
        issues.append(f"{no_salary} records have no valid salary (kept, but excluded from salary analysis).")

    # 7. Posted_Date -> datetime ---------------------------------------------
    raw_dates = df["Posted_Date"]
    try:
        parsed = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(raw_dates, errors="coerce")
    invalid_dates = int((parsed.isna() & raw_dates.notna()).sum())
    summary["invalid_posted_dates"] = invalid_dates
    df["Posted_Date"] = parsed
    if invalid_dates:
        issues.append(f"{invalid_dates} Posted_Date values could not be read and are left blank.")

    df = df.reset_index(drop=True)
    summary["records_after"] = len(df)
    summary["issues"] = issues
    return df, summary


def format_cleaning_summary(summary):
    """Return the cleaning summary as printable text."""
    lines = [
        "DATA CLEANING SUMMARY",
        "---------------------",
        f"Records before cleaning : {summary['records_before']}",
        f"Records after cleaning  : {summary['records_after']}",
        f"Duplicate Job_IDs removed       : {summary['duplicate_job_ids_removed']}",
        f"Records removed (no Job_ID)     : {summary['removed_missing_job_id']}",
        f"Records removed (no Job_Title)  : {summary['removed_missing_job_title']}",
        f"Text cells with spaces fixed    : {summary['cells_with_extra_spaces_fixed']}",
        f"Missing text filled             : {summary['missing_text_filled_with_not_specified']}",
        f"Non-numeric salary values       : {summary['non_numeric_salary_values']}",
        f"Records with negative salary    : {summary['records_with_negative_salary']}",
        f"Records with invalid salary range: {summary['records_with_invalid_salary_range']}",
        f"Records without valid salary    : {summary['records_without_valid_salary']}",
        f"Invalid Posted_Date values      : {summary['invalid_posted_dates']}",
    ]
    if summary["issues"]:
        lines.append("")
        lines.append("Data-quality warnings:")
        lines.extend(f"  ! {issue}" for issue in summary["issues"])
    else:
        lines.append("")
        lines.append("No data-quality problems were found.")
    return "\n".join(lines)
