"""
visualization.py
----------------
Matplotlib charts. Every function returns a matplotlib Figure so the same code
works in the console (plt.show) and in Streamlit (st.pyplot).
Empty data produces a figure with a clear "no data" message instead of an error.
"""

import matplotlib.pyplot as plt

import analysis

PALETTE = ["#2E86AB", "#F6AE2D", "#33A1FD", "#F26419", "#55A630", "#9D4EDD",
           "#E63946", "#06D6A0", "#8D99AE", "#BC6C25"]
NO_DATA_MESSAGE = "No data available for the selected filters."


def _no_data_figure(title):
    """Placeholder figure shown when there is nothing to plot."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, NO_DATA_MESSAGE, ha="center", va="center", fontsize=12, color="gray")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _legend_labels(counts_df, label_column):
    """Legend text like 'Data Analyst (12.5%)'."""
    return [f"{name} ({pct:.1f}%)"
            for name, pct in zip(counts_df[label_column], counts_df["Percent"])]


# 1. Horizontal bar chart - top skills ---------------------------------------
def plot_top_skills(df, top_n=10):
    title = f"Top {top_n} In-Demand Skills"
    skills = analysis.top_skills(df, top_n)
    if skills.empty:
        return _no_data_figure(title)
    skills = skills.iloc[::-1]  # largest bar at the top
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(skills) + 1.5)))
    bars = ax.barh(skills["Skill"], skills["Postings"], color=PALETTE[0])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Job Postings")
    ax.set_ylabel("Skill")
    ax.set_xlim(0, skills["Postings"].max() * 1.12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# 2. Donut chart - job role distribution ---------------------------------------
def plot_role_donut(df):
    title = "Job Role Distribution"
    roles = analysis.job_role_counts(df)
    if roles.empty:
        return _no_data_figure(title)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(roles))]
    wedges, _ = ax.pie(roles["Postings"], colors=colors, startangle=90, counterclock=False,
                       wedgeprops={"width": 0.42, "edgecolor": "white"})
    ax.text(0, 0, f"{len(df)}\npostings", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.legend(wedges, _legend_labels(roles, "Job_Title"), title="Job Role",
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("equal")
    fig.tight_layout()
    return fig


# 3. Bar chart - postings by location --------------------------------------------
def plot_location_bar(df):
    title = "Job Postings by Location"
    locations = analysis.location_counts(df)
    if locations.empty:
        return _no_data_figure(title)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(locations["Location"], locations["Postings"], color=PALETTE[3])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Location")
    ax.set_ylabel("Number of Job Postings")
    ax.set_ylim(0, locations["Postings"].max() * 1.12)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# 4. Line chart - salary comparison ----------------------------------------------
def plot_salary_line(df, by="experience"):
    """Salary line chart across experience levels (by='experience') or job roles (by='role')."""
    if by == "role":
        column, x_label, title = "Job_Title", "Job Role", "Average Salary by Job Role"
        data = analysis.salary_by_role(df)
    else:
        column, x_label, title = "Experience", "Experience Level", "Average Salary by Experience Level"
        data = analysis.salary_by_group(df, "Experience")
    if data.empty:
        return _no_data_figure(title)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data[column], data["Avg_Min_LPA"], marker="o", color=PALETTE[8], label="Average Minimum")
    ax.plot(data[column], data["Avg_Midpoint_LPA"], marker="o", linewidth=2.5,
            color=PALETTE[0], label="Average Midpoint")
    ax.plot(data[column], data["Avg_Max_LPA"], marker="o", color=PALETTE[3], label="Average Maximum")
    for x, y in zip(data[column], data["Avg_Midpoint_LPA"]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Salary (LPA)")
    ax.grid(alpha=0.3)
    ax.legend()
    if by == "role":
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


# 5. Pie chart - work modes --------------------------------------------------------
def plot_work_mode_pie(df):
    title = "Work Mode Distribution"
    modes = analysis.work_mode_counts(df)
    if modes.empty:
        return _no_data_figure(title)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(modes))]
    wedges, _, autotexts = ax.pie(modes["Postings"], colors=colors, autopct="%1.1f%%",
                                  startangle=90, counterclock=False,
                                  wedgeprops={"edgecolor": "white"})
    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")
    ax.legend(wedges, [f"{m} ({n})" for m, n in zip(modes["Work_Mode"], modes["Postings"])],
              title="Work Mode", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("equal")
    fig.tight_layout()
    return fig


# Extra - employment type bar (section H) ---------------------------------------------
def plot_employment_type_bar(df):
    title = "Employment Type Distribution"
    types = analysis.employment_type_counts(df)
    if types.empty:
        return _no_data_figure(title)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(types["Employment_Type"], types["Postings"], color=PALETTE[4])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Employment Type")
    ax.set_ylabel("Number of Job Postings")
    ax.set_ylim(0, types["Postings"].max() * 1.12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
