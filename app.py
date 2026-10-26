"""
app.py
------
Streamlit dashboard for the Job Market Skill Analysis System.
Run with:  streamlit run app.py

NOTE: The data is SYNTHETIC sample data, not verified live vacancies.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend required by Streamlit
import matplotlib.pyplot as plt
import streamlit as st

import analysis
import visualization
from preprocessing import format_cleaning_summary, load_dataset, preprocess_data

st.set_page_config(page_title="Job Market Skill Analysis System", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning data...")
def get_clean_data():
    """Load the CSV (generating the synthetic file if missing) and clean it."""
    raw = load_dataset()
    if raw is None:
        return None, None
    return preprocess_data(raw)


def show_figure(figure):
    """Display a matplotlib figure in Streamlit and free its memory."""
    st.pyplot(figure)
    plt.close(figure)


def show_table(table):
    if table.empty:
        st.info("No data available for the selected filters.")
    else:
        st.dataframe(table, hide_index=True)


# ------------------------------------------------------------------ header
st.title("📊 Job Market Skill Analysis System")
st.write("Explore in-demand skills, job roles, salaries, experience requirements and "
         "locations to understand what the job market asks for. Built for students and job seekers.")
st.warning("⚠️ **Synthetic sample data:** this dashboard uses a randomly generated academic dataset "
           "(2,500 records). It is **not real-time job-market data** and does not represent verified live vacancies.")

data, cleaning_summary = get_clean_data()
if data is None or data.empty:
    st.error("The dataset could not be loaded. Run `python data_generator.py` in the project folder "
             "and refresh this page. Details are shown in the terminal.")
    st.stop()

with st.expander("Data cleaning report (data-quality summary)"):
    st.text(format_cleaning_summary(cleaning_summary))

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
st.sidebar.caption("Leave a filter empty to include everything.")
selected_roles = st.sidebar.multiselect("Job role", sorted(data["Job_Title"].unique()))
selected_locations = st.sidebar.multiselect("Location", sorted(data["Location"].unique()))
selected_experience = st.sidebar.multiselect(
    "Experience", analysis.sort_experience(data["Experience"].unique()))
selected_employment = st.sidebar.multiselect("Employment type", sorted(data["Employment_Type"].unique()))
selected_modes = st.sidebar.multiselect("Work mode", sorted(data["Work_Mode"].unique()))

filtered = analysis.filter_data(data, selected_roles, selected_locations, selected_experience,
                                selected_employment, selected_modes)

# ------------------------------------------------------------------ KPI cards
salary_stats = analysis.salary_statistics(filtered)
average_salary = salary_stats["average_midpoint"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total job records", f"{len(filtered):,}")
col2.metric("Unique job roles", filtered["Job_Title"].nunique())
col3.metric("Unique locations", filtered["Location"].nunique())
col4.metric("Average salary", f"{average_salary:.2f} LPA" if average_salary is not None else "N/A")

if filtered.empty:
    st.warning("No job records match the selected filters. Please relax one or more filters.")

# ------------------------------------------------------------------ tabs
tab_skills, tab_roles, tab_salary, tab_exp, tab_fresher, tab_data = st.tabs(
    ["Skills", "Roles & Locations", "Salary", "Experience", "Fresher-Friendly Jobs", "Job Data"])

with tab_skills:
    st.subheader("Top in-demand skills")
    top_n = st.slider("Number of skills to display", 3, 20, 10)
    left, right = st.columns([3, 2])
    with left:
        show_figure(visualization.plot_top_skills(filtered, top_n))
    with right:
        show_table(analysis.top_skills(filtered, top_n))

    st.subheader("Skills required for a selected role")
    role_options = sorted(filtered["Job_Title"].unique())
    if role_options:
        chosen_role = st.selectbox("Choose a job role", role_options)
        role_skills = analysis.skills_for_role(filtered, chosen_role, top_n)
        st.caption(f"Most frequently requested skills in {chosen_role} postings.")
        show_table(role_skills)
    else:
        st.info("No roles available for the selected filters.")

with tab_roles:
    left, right = st.columns(2)
    with left:
        st.subheader("Job-role distribution")
        show_figure(visualization.plot_role_donut(filtered))
        show_table(analysis.job_role_counts(filtered))
    with right:
        st.subheader("Location-wise opportunities")
        show_figure(visualization.plot_location_bar(filtered))
        show_table(analysis.location_counts(filtered))

    left, right = st.columns(2)
    with left:
        st.subheader("Work mode distribution")
        show_figure(visualization.plot_work_mode_pie(filtered))
    with right:
        st.subheader("Employment type distribution")
        show_figure(visualization.plot_employment_type_bar(filtered))

with tab_salary:
    st.subheader("Salary analysis (LPA = lakhs per annum)")
    if salary_stats["records_used"] == 0:
        st.info("No records with valid salary for the selected filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average (midpoint)", f"{salary_stats['average_midpoint']} LPA")
        c2.metric("Median (midpoint)", f"{salary_stats['median_midpoint']} LPA")
        c3.metric("Lowest minimum", f"{salary_stats['lowest_salary']} LPA")
        c4.metric("Highest maximum", f"{salary_stats['highest_salary']} LPA")
        st.caption(f"Based on {salary_stats['records_used']} records with valid salary values.")
    compare_by = st.radio("Compare salaries across", ["Experience levels", "Job roles"], horizontal=True)
    show_figure(visualization.plot_salary_line(
        filtered, "experience" if compare_by == "Experience levels" else "role"))
    show_table(analysis.salary_by_group(
        filtered, "Experience" if compare_by == "Experience levels" else "Job_Title"))

with tab_exp:
    st.subheader("Experience analysis")
    left, right = st.columns(2)
    with left:
        st.write("**Postings and average salary by experience**")
        show_table(analysis.experience_summary(filtered))
    with right:
        st.write("**Freshers vs experienced candidates**")
        show_table(analysis.fresher_vs_experienced(filtered))

with tab_fresher:
    st.subheader("Fresher-friendly job listings")
    f1, f2 = st.columns(2)
    fresher_location = f1.selectbox(
        "Location", [analysis.ALL] + sorted(filtered["Location"].unique()))
    fresher_skill = f2.selectbox(
        "Skill", [analysis.ALL] + sorted(analysis.split_skills(filtered).unique()))
    fresher_table = analysis.fresher_jobs(filtered, fresher_location, fresher_skill)
    st.caption(f"{len(fresher_table)} fresher-friendly jobs found.")
    show_table(fresher_table)

with tab_data:
    st.subheader("Filtered job data")
    display = filtered.copy()
    display["Posted_Date"] = display["Posted_Date"].dt.strftime("%Y-%m-%d")
    st.caption(f"{len(display)} records (synthetic sample data).")
    show_table(display)
    st.download_button(
        label="⬇️ Download filtered results as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="filtered_job_market_data.csv",
        mime="text/csv",
        disabled=display.empty,
    )

st.caption("Job Market Skill Analysis System | Synthetic academic dataset - not live vacancies.")
