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
    return sum(len(re.findall(rf"\\b{re.escape(w)}\\b", text)) for w in flag_list)


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


st.title("NSF Grant Cancellations Dashboard")

##########################################################
# Q2 — Institutions by cancellations
##########################################################
institutions = terminated["org_name"].value_counts().reset_index()
institutions.columns = ["institution", "cancelled_grants"]

Q2 = (
    alt.Chart(institutions.head(15))
    .mark_bar()
    .encode(
        x=alt.X("cancelled_grants:Q", title="Cancelled grants"),
        y=alt.Y("institution:N", sort="-x"),
        color=alt.Color("cancelled_grants:Q", scale=alt.Scale(scheme="reds")),
    )
    .properties(width=600, height=350, title="Top 15 Institutions by Cancellations")
)


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

Q3 = (
    alt.Chart(budget.head(15))
    .mark_bar()
    .encode(
        x=alt.X("impact:Q", axis=alt.Axis(format="$,.0s"), title="Budget impact ($)"),
        y=alt.Y("org_name:N", sort="-x"),
        color=alt.Color("impact:Q", scale=alt.Scale(scheme="oranges")),
    )
    .properties(width=600, height=350, title="Top 15 Institutions by Budget Impact")
)


##########################################################
# Q4 — Flagged words distribution (corrected + resized)
##########################################################

# Work ONLY with cancelled grants
df_q4 = terminated.copy()

# ------------- LEFT GRAPH (frequency polygon)
vals = df_q4["flagged_words_count"].fillna(0).clip(upper=40)

bins = np.arange(0, 41, 1)
hist, edges = np.histogram(vals, bins=bins)

df_bins = pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "count": hist})
df_bins["mid"] = (df_bins["bin_start"] + df_bins["bin_end"]) / 2

chart_q4_left = (
    alt.Chart(df_bins)
    .mark_line(point=True)
    .encode(
        x=alt.X("mid:Q", title="Number of Flagged Words per Grant"),
        y=alt.Y("count:Q", title="Number of Grants"),
    )
    .properties(
        width=350,
        height=250,
        title="Distribution of Flagged Word Counts in Cancelled Grants",
    )
)


# ------------- RIGHT GRAPH (Top 15 flagged words)
pattern = r"\b(" + "|".join(re.escape(w) for w in flag_list) + r")\b"

word_counts = (
    df_q4["abstract"]
    .str.lower()
    .str.findall(pattern)  # ✔ WORD-LEVEL MATCHING
    .explode()
    .value_counts()
    .head(15)
    .reset_index()
)

word_counts.columns = ["word", "occurrences"]

chart_q4_right = (
    alt.Chart(word_counts)
    .mark_bar()
    .encode(
        x=alt.X("occurrences:Q", title="Occurrences"),
        y=alt.Y("word:N", sort="-x", title="Flagged Word"),
        color=alt.Color("occurrences:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["word:N", "occurrences:Q"],
    )
    .properties(
        width=350, height=250, title="Top 15 Flagged Words Found in Cancelled Grants"
    )
)


# ------------- FINAL Q4
Q4 = chart_q4_left | chart_q4_right


##########################################################
# Q5 — Cruz List vs Reinstatement (Advanced)
##########################################################

df_q5 = df.copy()

df_q5["cruz_label"] = df_q5["in_cruz_list"].map({True: "Yes", False: "No"})
df_q5["status_label"] = df_q5["reinstated"].map(
    {True: "Reinstated", False: "Terminated"}
)
df_q5["status_order"] = df_q5["status_label"].map({"Terminated": 0, "Reinstated": 1})

q5_counts = (
    df_q5.groupby(["cruz_label", "status_label", "status_order"])
    .size()
    .reset_index(name="count")
)

row_totals = (
    q5_counts.groupby("cruz_label")["count"].sum().reset_index(name="row_total")
)
q5_counts = q5_counts.merge(row_totals, on="cruz_label", how="left")
q5_counts["percentage"] = (q5_counts["count"] / q5_counts["row_total"] * 100).round(1)

# --- COLORS ---
BLUE_LIGHT = "#6fa8fa"
GRAY = "#a8a8a8"
DARK_TEXT = "#111827"

x_max = int(q5_counts["row_total"].max() * 1.1)
axis_values = list(range(0, x_max + 200, 200))

# --- Left panel ---
base_left = alt.Chart(q5_counts).properties(width=550, height=260)

stack = base_left.mark_bar().encode(
    y=alt.Y("cruz_label:N", title="In Cruz List", sort=["No", "Yes"]),
    x=alt.X("count:Q", title="Number of Grants", scale=alt.Scale(domain=[0, x_max])),
    color=alt.Color(
        "status_label:N",
        scale=alt.Scale(domain=["Terminated", "Reinstated"], range=[BLUE_LIGHT, GRAY]),
        legend=None,
    ),
    order=alt.Order("status_order:Q"),
)

# Add percent labels
percentage_labels = base_left.mark_text(
    fontSize=14, fontWeight=600, color="white", dx=5
).encode(x="count:Q", y="cruz_label:N", text="percentage:Q")

left_panel = stack + percentage_labels

# --- Right panel: overall totals ---
totals = (
    df_q5.groupby(["status_label", "status_order"])
    .size()
    .reset_index(name="count")
    .sort_values("status_order")
)
totals["one"] = "Totals"
total_sum = int(totals["count"].sum())
totals["percentage"] = (totals["count"] / total_sum * 100).round(1)

right_panel = (
    alt.Chart(totals)
    .mark_bar()
    .encode(
        x=alt.X("one:N", axis=None),
        y=alt.Y("count:Q", stack="zero", axis=None),
        color=alt.Color(
            "status_label:N",
            scale=alt.Scale(
                domain=["Terminated", "Reinstated"], range=[BLUE_LIGHT, GRAY]
            ),
            legend=None,
        ),
    )
    .properties(width=150, height=260)
)

right_labels = (
    alt.Chart(totals)
    .mark_text(fontSize=14, fontWeight=600, color="white")
    .encode(
        x="one:N",
        y=alt.Y("count:Q", stack="zero"),
        text=alt.Text("percentage:Q", format=".1f"),
    )
)

Q5 = left_panel | (right_panel + right_labels)

##########################################################
# LAYOUT
##########################################################

col1, col2 = st.columns(2)
with col1:
    st.altair_chart(Q2, use_container_width=True)
with col2:
    st.altair_chart(Q3, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.altair_chart(Q4, use_container_width=True)
with col4:
    st.altair_chart(Q5, use_container_width=True)
