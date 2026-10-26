# Job Market Skill Analysis System

A Python project that analyzes job-market data to find in-demand skills, popular roles, salary trends,
experience requirements, employment types and location-based opportunities. It has an interactive
**Streamlit dashboard** and an optional **console menu**.

> **Synthetic data notice:** the dataset (`job_market_skill_analysis_synthetic_2500.csv`, 2,500 records,
> 14 attributes) is randomly generated sample data for academic use. Company names are fictional.
> It is **not real-time job-market data** and does not represent verified live vacancies.

## Project structure

```
job_market_skill_analysis/
├── app.py                 Streamlit dashboard (filters, charts, tables, CSV download)
├── main.py                Console menu (10 options)
├── data_generator.py      Creates the synthetic CSV dataset locally
├── preprocessing.py       load_dataset() and preprocess_data() (loading + cleaning + cleaning report)
├── analysis.py            Analysis functions (skills, roles, locations, salary, experience, fresher jobs, ...)
├── visualization.py       Matplotlib charts (return Figure objects, handle empty data)
├── job_market_skill_analysis_synthetic_2500.csv   The dataset
├── requirements.txt       pandas, numpy, matplotlib, streamlit
└── README.md              This file
```

## Installation (Windows)

1. Install Python 3.9+ from https://www.python.org/downloads/ (tick **"Add Python to PATH"**).
2. Open **Command Prompt** or PowerShell and go to the project folder:

```
cd path\to\job_market_skill_analysis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running

| Task | Command |
|---|---|
| Generate / regenerate the dataset | `python data_generator.py` |
| Console application | `python main.py` |
| Dashboard | `streamlit run app.py` |

The dashboard opens automatically; if not, open **http://localhost:8501** in your browser.
Press `Ctrl+C` in the terminal to stop it. If the CSV is missing, both apps generate it
automatically (clearly labelled synthetic).

## Troubleshooting

* **`'python' is not recognized`** – reinstall Python with "Add Python to PATH", or use `py` instead of `python`.
* **`'streamlit' is not recognized`** – run `python -m streamlit run app.py`.
* **`ModuleNotFoundError`** – activate the virtual environment and run `pip install -r requirements.txt`;
  make sure you run commands from inside the project folder.
* **File-path errors / CSV not found** – run `python data_generator.py` from the project folder. The CSV is always
  looked up next to the Python files, so the working directory does not matter once the files are together.
* **Port already in use** – `streamlit run app.py --server.port 8502`.
* **Console graphs do not open** – they need a desktop session; the dashboard shows the same charts in the browser.

## Features

**Cleaning (`preprocess_data`)** – removes duplicate Job_IDs, trims spaces, handles missing values, converts salaries
to numbers, converts Posted_Date to dates, detects negative salaries and Min > Max ranges. Invalid salary values are set
to missing (the record is kept but ignored in salary statistics). Every problem is **counted and reported**
(console start-up warnings, option 1, and the dashboard's "Data cleaning report").

**Analysis** – top-N skills, role counts and role filter, location counts and location filter, salary
(average/median/min/max, by role and experience, filter by role/experience/location), experience groups and
fresher-vs-experienced comparison, fresher-friendly jobs (filter by location and skill), skills per role,
employment-type and work-mode distributions.

**Charts** – (1) horizontal bar: top skills, (2) donut: job roles, (3) bar: postings by location,
(4) line: salary by experience or role, (5) pie: work modes, plus an employment-type bar chart.

**Dashboard** – KPI cards, sidebar filters (role, location, experience, employment type, work mode; empty = all),
tabs for skills, roles & locations, salary, experience, fresher jobs and data table, and a CSV download of the
filtered results. Everything updates when filters change.

## Architecture and workflow

```
CSV  ->  load_dataset()  ->  preprocess_data()  ->  analysis.py  ->  visualization.py  ->  app.py / main.py
        (read + validate)   (clean + report)       (pandas maths)    (matplotlib)          (UI)
```

`data_generator.py` feeds the CSV; `preprocessing.py` depends on it only to create a missing CSV. `analysis.py` and
`visualization.py` never print or read input, so both interfaces share exactly the same logic.

## Sample output (real run on the included dataset)

```
Dataset loaded successfully: 2500 records, 14 attributes.
TOP 5 IN-DEMAND SKILLS
        Skill  Postings  Percent_of_Postings
          Git      1533                 61.3
Communication      1419                 56.8
       Python      1285                 51.4
        Linux      1080                 43.2
          SQL       990                 39.6

POSTINGS AND AVERAGE SALARY BY EXPERIENCE
Experience  Postings  Avg_Midpoint_LPA
 0-1 years       840              5.40
 1-3 years       705              9.69
 3-5 years       509             14.39
  5+ years       446             21.37
```

Explanation: Git, Communication and Python appear in more than half of the postings. Salary rises steadily with
experience (about 5.4 LPA for freshers to 21.4 LPA for 5+ years). Bengaluru (22.1%) and Hyderabad (21.3%) have the most
postings, and Machine Learning Engineer has the highest average salary. These results describe the synthetic data only.
