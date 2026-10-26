"""
main.py
-------
Console menu for the Job Market Skill Analysis System.
Run with:  python main.py

NOTE: The data is SYNTHETIC sample data - not real-time job-market data.
"""

import matplotlib.pyplot as plt
import pandas as pd

import analysis
import visualization
from preprocessing import format_cleaning_summary, load_dataset, preprocess_data

MENU = """
==============================================
   JOB MARKET SKILL ANALYSIS SYSTEM (console)
   Data: SYNTHETIC sample data, not live jobs
==============================================
 1. Display dataset information
 2. Show the top 10 skills
 3. Analyze job roles
 4. Analyze job locations
 5. Analyze salaries
 6. Analyze experience requirements
 7. Display fresher-friendly jobs
 8. Show skills required for a selected role
 9. Display graphs
10. Exit
"""


# ---------------------------- input helpers ----------------------------
def ask_int(prompt, default, minimum, maximum):
    """Ask for an integer in [minimum, maximum]; Enter keeps the default."""
    while True:
        text = input(f"{prompt} [default {default}]: ").strip()
        if text == "":
            return default
        try:
            value = int(text)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"  Please enter a number between {minimum} and {maximum}.")


def ask_choice(prompt, options):
    """Show a numbered list (0 = All) and return the chosen option or 'All'."""
    print(f"\n{prompt}")
    print("  0. All")
    for number, option in enumerate(options, start=1):
        print(f"  {number}. {option}")
    choice = ask_int("Enter your choice", 0, 0, len(options))
    return analysis.ALL if choice == 0 else options[choice - 1]


def show_table(table, max_rows=None):
    """Print a DataFrame without the index."""
    if table.empty:
        print("  (no data for this selection)")
        return
    if max_rows is not None and len(table) > max_rows:
        print(table.head(max_rows).to_string(index=False))
        print(f"  ... showing {max_rows} of {len(table)} rows")
    else:
        print(table.to_string(index=False))


# ---------------------------- menu actions ----------------------------
def show_dataset_info(df, summary):
    print("\nDATASET INFORMATION (SYNTHETIC sample data)")
    print(f"Total records     : {len(df)}")
    print(f"Attributes        : {len(df.columns)}")
    print(f"Unique job roles  : {df['Job_Title'].nunique()}")
    print(f"Unique locations  : {df['Location'].nunique()}")
    if df["Posted_Date"].notna().any():
        print(f"Posted date range : {df['Posted_Date'].min().date()} to {df['Posted_Date'].max().date()}")
    print("\nColumns:")
    print(", ".join(df.columns))
    print("\nFirst 5 records:")
    show_table(df.head().assign(Posted_Date=lambda d: d["Posted_Date"].dt.strftime("%Y-%m-%d")))
    print()
    print(format_cleaning_summary(summary))


def show_top_skills(df):
    count = ask_int("How many top skills do you want to see", 10, 1, 50)
    print(f"\nTOP {count} IN-DEMAND SKILLS")
    show_table(analysis.top_skills(df, count))


def show_role_analysis(df):
    print("\nJOB ROLES BY NUMBER OF POSTINGS")
    show_table(analysis.job_role_counts(df))
    roles = sorted(df["Job_Title"].unique())
    role = ask_choice("Select a job role to view its postings:", roles)
    if role != analysis.ALL:
        subset = analysis.filter_by_role(df, role)
        print(f"\n{role}: {len(subset)} postings. Sample (first 10):")
        show_table(subset[["Job_ID", "Company", "Location", "Experience", "Salary_Midpoint_LPA"]], 10)


def show_location_analysis(df):
    print("\nJOB POSTINGS BY LOCATION")
    show_table(analysis.location_counts(df))
    locations = sorted(df["Location"].unique())
    location = ask_choice("Select a location to view its postings:", locations)
    if location != analysis.ALL:
        subset = analysis.filter_by_location(df, location)
        print(f"\n{location}: {len(subset)} postings. Top roles:")
        show_table(analysis.job_role_counts(subset))


def show_salary_analysis(df):
    role = ask_choice("Filter by job role:", sorted(df["Job_Title"].unique()))
    experience = ask_choice("Filter by experience:", analysis.sort_experience(df["Experience"].unique()))
    location = ask_choice("Filter by location:", sorted(df["Location"].unique()))
    subset = analysis.filter_for_salary(df, role, experience, location)

    stats = analysis.salary_statistics(subset)
    print(f"\nSALARY ANALYSIS (role={role}, experience={experience}, location={location})")
    if stats["records_used"] == 0:
        print("  No records with valid salary for this selection.")
        return
    print(f"  Records with valid salary : {stats['records_used']}")
    print(f"  Average salary (midpoint) : {stats['average_midpoint']} LPA")
    print(f"  Median salary (midpoint)  : {stats['median_midpoint']} LPA")
    print(f"  Lowest minimum salary     : {stats['lowest_salary']} LPA")
    print(f"  Highest maximum salary    : {stats['highest_salary']} LPA")
    print("\nSalary comparison across job roles (LPA):")
    show_table(analysis.salary_by_role(subset))


def show_experience_analysis(df):
    print("\nPOSTINGS AND AVERAGE SALARY BY EXPERIENCE")
    show_table(analysis.experience_summary(df))
    print("\nFRESHERS vs EXPERIENCED")
    show_table(analysis.fresher_vs_experienced(df))


def show_fresher_jobs(df):
    location = ask_choice("Filter by location:", sorted(df["Location"].unique()))
    skills = sorted(analysis.split_skills(df).unique())
    skill = ask_choice("Filter by skill:", skills)
    jobs = analysis.fresher_jobs(df, location, skill)
    print(f"\nFRESHER-FRIENDLY JOBS ({len(jobs)} found, location={location}, skill={skill})")
    show_table(jobs, 20)


def show_role_skills(df):
    roles = sorted(df["Job_Title"].unique())
    print("\nSelect a job role:")
    for number, role in enumerate(roles, start=1):
        print(f"  {number}. {role}")
    role = roles[ask_int("Enter your choice", 1, 1, len(roles)) - 1]
    count = ask_int("How many skills to display", 10, 1, 50)
    print(f"\nTOP SKILLS FOR: {role}")
    show_table(analysis.skills_for_role(df, role, count))


def show_graphs(df):
    print("\nOpening the 5 required graphs plus an employment-type chart. Close the windows to return to the menu.")
    figures = [
        visualization.plot_top_skills(df, 10),
        visualization.plot_role_donut(df),
        visualization.plot_location_bar(df),
        visualization.plot_salary_line(df, "experience"),
        visualization.plot_work_mode_pie(df),
        visualization.plot_employment_type_bar(df),
    ]
    print(f"({len(figures)} figures ready)")
    try:
        plt.show()
    except Exception as error:  # e.g. no display available
        print(f"Could not open the graph windows: {error}")
    finally:
        plt.close("all")


# ---------------------------- program flow ----------------------------
def prepare_data():
    """Load and clean the data. Returns (df, summary) or (None, None)."""
    raw = load_dataset()
    if raw is None:
        return None, None
    df, summary = preprocess_data(raw)
    if summary["issues"]:
        print("\nData-quality warnings found while cleaning:")
        for issue in summary["issues"]:
            print(f"  ! {issue}")
    if df.empty:
        print("ERROR: No valid records remain after cleaning.")
        return None, None
    return df, summary


def main():
    df, summary = prepare_data()
    if df is None:
        return
    actions = {
        1: lambda: show_dataset_info(df, summary),
        2: lambda: show_top_skills(df),
        3: lambda: show_role_analysis(df),
        4: lambda: show_location_analysis(df),
        5: lambda: show_salary_analysis(df),
        6: lambda: show_experience_analysis(df),
        7: lambda: show_fresher_jobs(df),
        8: lambda: show_role_skills(df),
        9: lambda: show_graphs(df),
    }
    while True:
        print(MENU)
        try:
            choice = input("Enter your choice (1-10): ").strip()
            if choice == "10":
                print("Goodbye!")
                break
            if not choice.isdigit() or int(choice) not in actions:
                print("Invalid choice. Please enter a number from 1 to 10.")
                continue
            actions[int(choice)]()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
        except Exception as error:  # keep the menu alive on unexpected errors
            print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
