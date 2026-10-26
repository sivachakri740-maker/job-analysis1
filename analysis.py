"""
analysis.py
-----------
All data-analysis functions. They take a cleaned DataFrame and return pandas
objects (DataFrame / Series / dict), so both the console and the dashboard can use them.
Missing or invalid salary values (NaN) are ignored in salary calculations.
"""

import pandas as pd

ALL = "All"  # value meaning "no filter"
EXPERIENCE_ORDER = ["0-1 years", "1-3 years", "3-5 years", "5+ years"]
FRESHER_EXPERIENCE = "0-1 years"


def sort_experience(values):
    """Sort experience labels in a natural order (unknown labels go last)."""
    values = list(values)
    known = [v for v in EXPERIENCE_ORDER if v in values]
    others = sorted(v for v in values if v not in EXPERIENCE_ORDER)
    return known + others


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def filter_data(df, roles=None, locations=None, experiences=None,
                employment_types=None, work_modes=None):
    """Keep rows matching every selected filter. None or an empty list means 'no filter'."""
    result = df
    for column, selected in [
        ("Job_Title", roles), ("Location", locations), ("Experience", experiences),
        ("Employment_Type", employment_types), ("Work_Mode", work_modes),
    ]:
        if selected:
            result = result[result[column].isin(selected)]
    return result


def _select(df, column, value):
    """Filter one column unless value is None / 'All'."""
    if value is None or value == ALL:
        return df
    return df[df[column] == value]


# --------------------------------------------------------------------------
# A. Skills
# --------------------------------------------------------------------------
def split_skills(df):
    """Return a Series with one skill per row (comma-separated skills are split)."""
    if df.empty:
        return pd.Series([], dtype="object")
    skills = df["Skills"].dropna().str.split(",").explode().str.strip()
    return skills[(skills != "") & (skills != "Not Specified")]


def top_skills(df, top_n=10):
    """Most requested skills: columns Skill, Postings, Percent_of_Postings."""
    skills = split_skills(df)
    if skills.empty:
        return pd.DataFrame(columns=["Skill", "Postings", "Percent_of_Postings"])
    counts = skills.value_counts().head(top_n)
    result = counts.rename_axis("Skill").reset_index(name="Postings")
    result["Percent_of_Postings"] = (result["Postings"] / len(df) * 100).round(1)
    return result


def skills_for_role(df, role, top_n=10):
    """Most requested skills for one job title."""
    return top_skills(df[df["Job_Title"] == role], top_n)


# --------------------------------------------------------------------------
# B / C. Roles and locations
# --------------------------------------------------------------------------
def _count_table(df, column, label, top_n=None):
    counts = df[column].value_counts()
    if top_n:
        counts = counts.head(top_n)
    result = counts.rename_axis(label).reset_index(name="Postings")
    if len(df):
        result["Percent"] = (result["Postings"] / len(df) * 100).round(1)
    else:
        result["Percent"] = []
    return result


def job_role_counts(df, top_n=None):
    """Number of postings per Job_Title."""
    return _count_table(df, "Job_Title", "Job_Title", top_n)


def location_counts(df, top_n=None):
    """Number of postings per city."""
    return _count_table(df, "Location", "Location", top_n)


def filter_by_role(df, role):
    return _select(df, "Job_Title", role)


def filter_by_location(df, location):
    return _select(df, "Location", location)


# --------------------------------------------------------------------------
# D. Salary
# --------------------------------------------------------------------------
def valid_salary_rows(df):
    """Rows whose min, max and midpoint salaries are all valid numbers."""
    salary_cols = ["Salary_Min_LPA", "Salary_Max_LPA", "Salary_Midpoint_LPA"]
    valid = df.dropna(subset=salary_cols)
    return valid[(valid[salary_cols] >= 0).all(axis=1)]


def salary_statistics(df):
    """Average, minimum and maximum salary (LPA) for the given rows."""
    valid = valid_salary_rows(df)
    if valid.empty:
        return {"records_used": 0, "average_midpoint": None, "median_midpoint": None,
                "lowest_salary": None, "highest_salary": None,
                "average_min": None, "average_max": None}
    return {
        "records_used": int(len(valid)),
        "average_midpoint": round(float(valid["Salary_Midpoint_LPA"].mean()), 2),
        "median_midpoint": round(float(valid["Salary_Midpoint_LPA"].median()), 2),
        "lowest_salary": round(float(valid["Salary_Min_LPA"].min()), 2),
        "highest_salary": round(float(valid["Salary_Max_LPA"].max()), 2),
        "average_min": round(float(valid["Salary_Min_LPA"].mean()), 2),
        "average_max": round(float(valid["Salary_Max_LPA"].mean()), 2),
    }


def filter_for_salary(df, role=None, experience=None, location=None):
    """Filter by role, experience and location (None or 'All' = no filter)."""
    result = _select(df, "Job_Title", role)
    result = _select(result, "Experience", experience)
    return _select(result, "Location", location)


def salary_by_group(df, column):
    """Salary comparison grouped by a column (e.g. Job_Title or Experience)."""
    columns = [column, "Postings", "Avg_Min_LPA", "Avg_Midpoint_LPA", "Avg_Max_LPA"]
    valid = valid_salary_rows(df)
    if valid.empty:
        return pd.DataFrame(columns=columns)
    grouped = valid.groupby(column).agg(
        Postings=("Job_ID", "count"),
        Avg_Min_LPA=("Salary_Min_LPA", "mean"),
        Avg_Midpoint_LPA=("Salary_Midpoint_LPA", "mean"),
        Avg_Max_LPA=("Salary_Max_LPA", "mean"),
    ).round(2).reset_index()
    if column == "Experience":
        order = sort_experience(grouped[column])
        grouped[column] = pd.Categorical(grouped[column], categories=order, ordered=True)
        grouped = grouped.sort_values(column).reset_index(drop=True)
        grouped[column] = grouped[column].astype(str)
    else:
        grouped = grouped.sort_values("Avg_Midpoint_LPA", ascending=False).reset_index(drop=True)
    return grouped


def salary_by_role(df):
    return salary_by_group(df, "Job_Title")


# --------------------------------------------------------------------------
# E. Experience
# --------------------------------------------------------------------------
def experience_counts(df):
    """Postings per experience band, in natural order."""
    counts = df["Experience"].value_counts()
    ordered = sort_experience(counts.index)
    result = counts.reindex(ordered).rename_axis("Experience").reset_index(name="Postings")
    return result


def experience_summary(df):
    """Postings and average salary per experience band."""
    counts = experience_counts(df)
    salary = salary_by_group(df, "Experience")
    if counts.empty:
        return counts
    merged = counts.merge(salary[["Experience", "Avg_Midpoint_LPA"]], on="Experience", how="left")
    return merged


def fresher_vs_experienced(df):
    """Compare fresher-level (0-1 years) postings with experienced postings."""
    rows = []
    groups = [("Fresher (0-1 years)", df[df["Experience"] == FRESHER_EXPERIENCE]),
              ("Experienced (1+ years)", df[df["Experience"] != FRESHER_EXPERIENCE])]
    for label, subset in groups:
        stats = salary_statistics(subset)
        rows.append({"Group": label, "Postings": len(subset),
                     "Avg_Midpoint_LPA": stats["average_midpoint"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# F. Fresher-friendly jobs
# --------------------------------------------------------------------------
def fresher_jobs(df, location=None, skill=None):
    """Fresher-friendly jobs, optionally filtered by location and skill."""
    jobs = df[df["Fresher_Friendly"] == "Yes"]
    jobs = _select(jobs, "Location", location)
    if skill and skill != ALL:
        wanted = skill.strip().lower()
        jobs = jobs[jobs["Skills"].map(
            lambda text: wanted in [s.strip().lower() for s in str(text).split(",")])]
    result = jobs[["Job_Title", "Company", "Location", "Skills",
                   "Salary_Min_LPA", "Salary_Max_LPA", "Employment_Type"]].copy()
    result["Salary_Range_LPA"] = result.apply(
        lambda r: f"{r['Salary_Min_LPA']:.2f} - {r['Salary_Max_LPA']:.2f}"
        if pd.notna(r["Salary_Min_LPA"]) and pd.notna(r["Salary_Max_LPA"]) else "N/A", axis=1)
    return result.drop(columns=["Salary_Min_LPA", "Salary_Max_LPA"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# H. Employment type and work mode
# --------------------------------------------------------------------------
def employment_type_counts(df):
    return _count_table(df, "Employment_Type", "Employment_Type")


def work_mode_counts(df):
    return _count_table(df, "Work_Mode", "Work_Mode")
