"""
data_generator.py
-----------------
Creates the SYNTHETIC job-market dataset used by the project.

IMPORTANT: Every record produced here is randomly generated sample data.
It is NOT real-time job-market data and does not describe verified vacancies.
Company names are fictional.

Run directly to (re)create the CSV file:
    python data_generator.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# The CSV is always stored next to this script so the project works from any folder.
DATASET_FILENAME = "job_market_skill_analysis_synthetic_2500.csv"
DATASET_PATH = Path(__file__).resolve().parent / DATASET_FILENAME

NUM_RECORDS = 2500
RANDOM_SEED = 42

# The 14 attributes required by the project, in column order.
COLUMNS = [
    "Job_ID", "Job_Title", "Company", "Location", "Skills", "Experience",
    "Salary_Min_LPA", "Salary_Max_LPA", "Salary_Midpoint_LPA",
    "Employment_Type", "Work_Mode", "Education", "Fresher_Friendly", "Posted_Date",
]

# Role -> required ("core") skills, extra ("optional") skills, and a base
# mid-point salary (LPA) for a candidate with 0-1 years of experience.
ROLE_PROFILES = {
    "Python Developer": {
        "core": ["Python", "Git"],
        "optional": ["SQL", "Linux", "AWS", "Data Analysis", "Machine Learning", "Communication"],
        "base_salary": 4.5,
    },
    "Data Analyst": {
        "core": ["SQL", "Excel", "Data Analysis"],
        "optional": ["Python", "Power BI", "Communication", "Git"],
        "base_salary": 4.2,
    },
    "Data Scientist": {
        "core": ["Python", "Machine Learning", "Data Analysis"],
        "optional": ["SQL", "TensorFlow", "Git", "AWS", "Power BI", "Communication"],
        "base_salary": 7.0,
    },
    "Software Engineer": {
        "core": ["Git"],
        "optional": ["Java", "Python", "C++", "SQL", "Linux", "AWS", "Communication"],
        "base_salary": 5.0,
    },
    "Web Developer": {
        "core": ["JavaScript", "HTML/CSS"],
        "optional": ["React", "Git", "SQL", "Python", "Communication"],
        "base_salary": 4.0,
    },
    "Machine Learning Engineer": {
        "core": ["Python", "Machine Learning", "TensorFlow"],
        "optional": ["Git", "AWS", "Linux", "SQL", "Data Analysis"],
        "base_salary": 8.0,
    },
    "Java Developer": {
        "core": ["Java", "SQL", "Git"],
        "optional": ["Linux", "AWS", "C++", "Communication"],
        "base_salary": 4.5,
    },
    "Cloud Engineer": {
        "core": ["AWS", "Linux"],
        "optional": ["Python", "Git", "Docker", "SQL", "Communication"],
        "base_salary": 6.0,
    },
    "Cybersecurity Analyst": {
        "core": ["Linux", "Network Security"],
        "optional": ["Python", "Communication", "AWS", "Git"],
        "base_salary": 5.5,
    },
    "DevOps Engineer": {
        "core": ["Linux", "Git", "Docker"],
        "optional": ["AWS", "Python", "Kubernetes", "Communication"],
        "base_salary": 6.5,
    },
}

# Location -> salary factor (metro cities pay slightly more in this sample).
LOCATION_FACTORS = {
    "Bengaluru": 1.15, "Hyderabad": 1.10, "Mumbai": 1.10, "Pune": 1.05,
    "Delhi": 1.05, "Chennai": 1.00, "Visakhapatnam": 0.88,
}
LOCATION_WEIGHTS = [0.22, 0.22, 0.14, 0.14, 0.12, 0.10, 0.06]

# Experience band -> salary multiplier.
EXPERIENCE_FACTORS = {"0-1 years": 1.0, "1-3 years": 1.6, "3-5 years": 2.4, "5+ years": 3.5}
EXPERIENCE_WEIGHTS = [0.28, 0.30, 0.22, 0.20]

EMPLOYMENT_FACTORS = {"Full-time": 1.0, "Part-time": 0.6, "Internship": 0.4, "Contract": 1.1}
EMPLOYMENT_WEIGHTS = [0.78, 0.05, 0.07, 0.10]

WORK_MODES = ["On-site", "Hybrid", "Remote"]
WORK_MODE_WEIGHTS = [0.45, 0.35, 0.20]

EDUCATION = ["B.Tech/B.E.", "BCA", "B.Sc", "MCA", "M.Tech"]
EDUCATION_WEIGHTS = [0.50, 0.10, 0.12, 0.12, 0.16]

# Fictional company names (no real employers are represented).
COMPANIES = [
    "Nimbus Technologies", "BlueOrbit Systems", "Quantum Leaf Labs", "Apex Digital",
    "Zenith Software", "CodeCraft Solutions", "DataBridge Analytics", "Skyline Infotech",
    "PixelForge Studios", "Vertex Cloudworks", "Lumen Data Works", "Orion Softech",
    "BrightPath Systems", "CyberNest Security", "Innovara Labs", "TerraByte Technologies",
    "Helix Informatics", "Cobalt Networks", "Silverline IT Services", "NovaStack Solutions",
    "Pioneer Datatech", "Radiant Software", "Everest Analytics", "Mosaic Technologies",
    "Catalyst Infosystems", "Aurora Cloud Services", "Fusion Digital Works", "Summit Codeworks",
]

# Posted dates are spread over the 365 days before this fixed date (reproducible output).
LATEST_POSTED_DATE = pd.Timestamp("2026-10-01")


def _pick_skills(rng, profile):
    """Return a comma-separated skill string: core skills + a few optional ones."""
    core = profile["core"]
    optional = profile["optional"]
    # Keep 2-3 core skills (or all of them when the role has fewer).
    n_core = min(len(core), int(rng.integers(2, 4)))
    chosen = list(rng.choice(core, size=n_core, replace=False))
    # Add 1-3 optional skills.
    n_opt = min(len(optional), int(rng.integers(1, 4)))
    chosen += list(rng.choice(optional, size=n_opt, replace=False))
    # Communication is commonly requested across roles.
    if "Communication" not in chosen and rng.random() < 0.30:
        chosen.append("Communication")
    return ", ".join(chosen)


def generate_dataset(num_records=NUM_RECORDS, seed=RANDOM_SEED):
    """Generate the synthetic dataset and return it as a pandas DataFrame."""
    rng = np.random.default_rng(seed)
    roles = list(ROLE_PROFILES)
    locations = list(LOCATION_FACTORS)
    experiences = list(EXPERIENCE_FACTORS)
    employment_types = list(EMPLOYMENT_FACTORS)

    rows = []
    for i in range(1, num_records + 1):
        title = str(rng.choice(roles))
        profile = ROLE_PROFILES[title]
        location = str(rng.choice(locations, p=LOCATION_WEIGHTS))
        experience = str(rng.choice(experiences, p=EXPERIENCE_WEIGHTS))
        employment = str(rng.choice(employment_types, p=EMPLOYMENT_WEIGHTS))
        # Internships are always entry level.
        if employment == "Internship":
            experience = "0-1 years"

        # Salary: base x experience x location x employment type x random variation.
        midpoint = (
            profile["base_salary"]
            * EXPERIENCE_FACTORS[experience]
            * LOCATION_FACTORS[location]
            * EMPLOYMENT_FACTORS[employment]
            * rng.uniform(0.85, 1.20)
        )
        salary_min = round(midpoint * rng.uniform(0.80, 0.92), 2)
        salary_max = round(midpoint * rng.uniform(1.08, 1.25), 2)
        salary_mid = round((salary_min + salary_max) / 2, 2)

        # Fresher friendly: entry-level jobs (almost always) and all internships.
        if employment == "Internship":
            fresher = "Yes"
        elif experience == "0-1 years" and rng.random() < 0.85:
            fresher = "Yes"
        else:
            fresher = "No"

        posted = LATEST_POSTED_DATE - pd.Timedelta(days=int(rng.integers(0, 365)))

        rows.append({
            "Job_ID": f"JOB{i:04d}",
            "Job_Title": title,
            "Company": str(rng.choice(COMPANIES)),
            "Location": location,
            "Skills": _pick_skills(rng, profile),
            "Experience": experience,
            "Salary_Min_LPA": salary_min,
            "Salary_Max_LPA": salary_max,
            "Salary_Midpoint_LPA": salary_mid,
            "Employment_Type": employment,
            "Work_Mode": str(rng.choice(WORK_MODES, p=WORK_MODE_WEIGHTS)),
            "Education": str(rng.choice(EDUCATION, p=EDUCATION_WEIGHTS)),
            "Fresher_Friendly": fresher,
            "Posted_Date": posted.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(rows, columns=COLUMNS)


def save_dataset(path=DATASET_PATH, num_records=NUM_RECORDS, seed=RANDOM_SEED):
    """Generate the dataset and write it to CSV. Returns the DataFrame."""
    data = generate_dataset(num_records, seed)
    data.to_csv(path, index=False, encoding="utf-8")
    return data


if __name__ == "__main__":
    dataset = save_dataset()
    print("SYNTHETIC SAMPLE DATA - not real-time job-market data.")
    print(f"Created {len(dataset)} records with {len(dataset.columns)} attributes.")
    print(f"Saved to: {DATASET_PATH}")
