import app as st
import pandas as pd
import numpy as np
import re
import altair as alt

st.set_page_config(page_title="NSF Terminations Dashboard", layout="wide")


##########################################################
# 1. LOAD DATA
##########################################################
@st.cache_data
def load_data():
    nsf = pd.read_csv("nsf_terminations_airtable.csv")
    cruz = pd.read_csv("cruz_list.csv", sep=";")
    flagged = pd.read_csv("flagged_words_trump_admin.csv")
    return nsf, cruz, flagged


nsf_data, cruz_data, flagged_words = load_data()

##########################################################
# 2. CLEAN DATA
##########################################################
columns_to_remove = [
    "usa_start_date",
    "usa_end_date",
    "nsf_start_date",
    "nsf_end_date",
    "status",
    "suspended",
    "nsf_url",
    "usaspending_url",
    "org_city",
    "award_type",
    "nsf_primary_program",
    "record_sha1",
]

df = nsf_data.drop(columns=columns_to_remove, errors="ignore")
df["termination_date"] = pd.to_datetime(df["termination_date"], errors="coerce")
df["terminated"] = df["terminated"].astype(bool)
df["reinstated"] = df["reinstated"].astype(bool)

numeric_cols = [
    "nsf_total_budget",
    "nsf_obligated",
    "usaspending_obligated",
    "usaspending_outlaid",
    "estimated_budget",
    "estimated_outlays",
    "estimated_remaining",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

##########################################################
# 3. FLAGGED WORDS
##########################################################
flag_list = [str(w).lower().strip(", ") for w in flagged_words["flagged_word"]]


def count_flagged(text):
    if pd.isna(text):
        return 0
    text = text.lower()
    return sum(len(re.findall(r"\b{}\b".format(re.escape(w)), text)) for w in flag_list)


df["flagged_words_count"] = df["abstract"].apply(count_flagged)

##########################################################
# 4. MERGE CRUZ LIST
##########################################################
cruz = cruz_data.rename(columns={"grant_number": "grant_id"})
df = df.merge(cruz[["grant_id", "in_cruz_list"]], on="grant_id", how="left")
df["in_cruz_list"] = df["in_cruz_list"].fillna(False)

##########################################################
# 5. PREPARE DATA
##########################################################
terminated = df[df["terminated"]].copy()

##########################################################
# TITLE
##########################################################
st.title("NSF Grant Cancellations Dashboard — Testing Charts")

##########################################################
# Q2 — Institutions by cancellations
##########################################################
institutions = terminated["org_name"].value_counts().reset_index()
institutions.columns = ["institution", "cancelled_grants"]

chart_q2 = (
    alt.Chart(institutions.head(15))
    .mark_bar()
    .encode(
        x=alt.X("cancelled_grants:Q", title="Cancelled grants"),
        y=alt.Y("institution:N", sort="-x", title="Institution"),
        color=alt.Color(
            "cancelled_grants:Q", scale=alt.Scale(scheme="reds"), legend=None
        ),
        tooltip=["institution", "cancelled_grants"],
    )
    .properties(
        width=500, height=350, title="Q2 — Top 15 Institutions by Cancellations"
    )
)

st.subheader("Q2 – Institutions most affected (count)")
st.altair_chart(chart_q2, use_container_width=True)

##########################################################
# Q3 — Budget impact
##########################################################
budget = (
    terminated.groupby("org_name")
    .agg({"nsf_total_budget": "sum", "estimated_budget": "sum"})
    .reset_index()
)
budget["impact"] = budget["nsf_total_budget"].fillna(budget["estimated_budget"])
budget = budget[budget["impact"] > 0].sort_values("impact", ascending=False)

chart_q3 = (
    alt.Chart(budget.head(15))
    .mark_bar()
    .encode(
        x=alt.X("impact:Q", axis=alt.Axis(format="$,.0s"), title="Budget impact ($)"),
        y=alt.Y("org_name:N", sort="-x", title="Institution"),
        color=alt.Color("impact:Q", scale=alt.Scale(scheme="oranges"), legend=None),
        tooltip=[
            alt.Tooltip("org_name", title="Institution"),
            alt.Tooltip("impact:Q", title="Budget impact", format="$,.0f"),
        ],
    )
    .properties(
        width=500, height=350, title="Q3 — Top 15 Institutions by Budget Impact"
    )
)

st.subheader("Q3 – Institutions most affected (budget)")
st.altair_chart(chart_q3, use_container_width=True)

##########################################################
# Q4 — Flagged words distribution
##########################################################
vals = terminated["flagged_words_count"].fillna(0).clip(upper=40)
hist, edges = np.histogram(vals, bins=np.arange(0, 41, 1))
df_bins = pd.DataFrame({"mid": (edges[:-1] + edges[1:]) / 2, "count": hist})

chart_q4 = (
    alt.Chart(df_bins)
    .mark_line(point=True)
    .encode(
        x=alt.X("mid:Q", title="Flagged words per grant"),
        y=alt.Y("count:Q", title="Number of grants"),
        tooltip=["mid", "count"],
    )
    .properties(width=500, height=350, title="Q4 — Flagged Words Distribution")
)

st.subheader("Q4 – Flagged language distribution")
st.altair_chart(chart_q4, use_container_width=True)

##########################################################
# Q5 — Cruz list vs status
##########################################################
df["cruz_label"] = df["in_cruz_list"].map({True: "Yes", False: "No"})
df["status_label"] = df["reinstated"].map({True: "Reinstated", False: "Terminated"})
df["status_order"] = df["status_label"].map({"Terminated": 0, "Reinstated": 1})

q5_counts = (
    df.groupby(["cruz_label", "status_label", "status_order"])
    .size()
    .reset_index(name="count")
)
q5_counts["row_total"] = q5_counts.groupby("cruz_label")["count"].transform("sum")
q5_counts["percentage"] = (q5_counts["count"] / q5_counts["row_total"] * 100).round(1)

BLUE_LIGHT = "#A8D5E2"
BLUE_MED = "#5B9BD5"

chart_q5 = (
    alt.Chart(q5_counts)
    .mark_bar()
    .encode(
        y=alt.Y("cruz_label:N", sort=["No", "Yes"], title="In Cruz list"),
        x=alt.X("count:Q", title="Number of grants"),
        color=alt.Color(
            "status_label:N",
            scale=alt.Scale(range=[BLUE_MED, BLUE_LIGHT]),
            legend=alt.Legend(title="Status"),
        ),
        tooltip=["cruz_label", "status_label", "count", "percentage"],
    )
    .properties(width=500, height=250, title="Q5 — Cruz List Status Comparison")
)

st.subheader("Q5 – Cruz list vs reinstatement")
st.altair_chart(chart_q5, use_container_width=True)
