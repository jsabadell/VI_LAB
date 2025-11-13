import streamlit as st  
import altair as alt
import pandas as pd
from vega_datasets import data


# Optional: avoid Altair's 5k-row limit
alt.data_transformers.disable_max_rows()

# === Load all datasets (same as notebook) ===
nsf_data = pd.read_csv("data/raw/nsf_terminations_airtable_copy.csv")
cruz_data = pd.read_csv("data/raw/cruz_list_copy.csv", sep=";")
flagged_words = pd.read_csv("data/raw/flagged_words_trump_admin_copy.csv")

columns_to_remove = [
    "usa_start_date", "usa_end_date", "nsf_start_date", "nsf_end_date",
    "status", "suspended", "nsf_url", "usaspending_url",
    "org_city", "award_type", "nsf_primary_program", "record_sha1"
]

cleaned_nsf_data = nsf_data.drop(columns=columns_to_remove, errors="ignore")

# dates
if "termination_date" in cleaned_nsf_data.columns:
    cleaned_nsf_data["termination_date"] = pd.to_datetime(
        cleaned_nsf_data["termination_date"], errors="coerce"
    )

# booleans – adapt if you have a couple more in your notebook
bool_cols = ["terminated", "reinstated", "in_cruz_list"]
for col in bool_cols:
    if col in cleaned_nsf_data.columns:
        cleaned_nsf_data[col] = cleaned_nsf_data[col].astype(bool)

# numerics
numeric_columns = [
    "nsf_total_budget", "nsf_obligated", "usaspending_obligated",
    "usaspending_outlaid", "estimated_budget",
    "estimated_outlays", "estimated_remaining"
]
for col in numeric_columns:
    if col in cleaned_nsf_data.columns:
        cleaned_nsf_data[col] = pd.to_numeric(
            cleaned_nsf_data[col], errors="coerce"
        )

# convenience subset used in several questions
terminated_grants = cleaned_nsf_data[cleaned_nsf_data["terminated"] == True]

# ---------------------

# Q1 data prep – cancellations by state
terminated_grants = cleaned_nsf_data[cleaned_nsf_data['terminated'] == True]

state_cancellations = (
    terminated_grants['org_state'].value_counts().reset_index()
)
state_cancellations.columns = ['state', 'cancelled_grants']

import re

# === Prepare flagged words and add counts to cleaned_nsf_data ===
flagged_words_clean = [
    str(w).strip().lower().strip(",")
    for w in flagged_words["flagged_word"]
]

def count_flagged_words(text, words):
    """Count occurrences of flagged words in a text (word-boundary aware)."""
    if pd.isna(text):
        return 0
    t = str(text).lower()
    count = 0
    for w in words:
        # count how many times this word appears as a whole word
        matches = re.findall(rf"\b{re.escape(w)}\b", t)
        count += len(matches)
    return count

# Add flagged word counts to the main dataset
if "abstract" in cleaned_nsf_data.columns:
    cleaned_nsf_data["flagged_words_count"] = cleaned_nsf_data["abstract"].apply(
        lambda x: count_flagged_words(x, flagged_words_clean)
    )

if "project_title" in cleaned_nsf_data.columns:
    cleaned_nsf_data["title_flagged_words_count"] = cleaned_nsf_data["project_title"].apply(
        lambda x: count_flagged_words(x, flagged_words_clean)
    )


import numpy as np

# === Merge Cruz list into main NSF dataset ===
if "grant_id" in cleaned_nsf_data.columns and "grant_number" in cruz_data.columns:
    cruz_renamed = cruz_data.rename(columns={"grant_number": "grant_id"})
    cleaned_nsf_data = cleaned_nsf_data.merge(
        cruz_renamed[["grant_id", "in_cruz_list"]],
        on="grant_id",
        how="left"
    )

# If merge failed for some reason, create a safe default column
if "in_cruz_list" not in cleaned_nsf_data.columns:
    cleaned_nsf_data["in_cruz_list"] = False

# Normalize in_cruz_list: fill NaNs and cast to bool
cleaned_nsf_data["in_cruz_list"] = (
    cleaned_nsf_data["in_cruz_list"]
    .fillna(False)
    .astype(bool)
)





# Q1
import altair as alt
from vega_datasets import data

# Load US topojson
states = alt.topo_feature(data.us_10m.url, "states")

# FIPS mapping
state_fips = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9, "DE": 10, "DC": 11,
    "FL": 12, "GA": 13, "HI": 15, "ID": 16, "IL": 17, "IN": 18, "IA": 19, "KS": 20,
    "KY": 21, "LA": 22, "ME": 23, "MD": 24, "MA": 25, "MI": 26, "MN": 27, "MS": 28,
    "MO": 29, "MT": 30, "NE": 31, "NV": 32, "NH": 33, "NJ": 34, "NM": 35, "NY": 36,
    "NC": 37, "ND": 38, "OH": 39, "OK": 40, "OR": 41, "PA": 42, "RI": 44, "SC": 45,
    "SD": 46, "TN": 47, "TX": 48, "UT": 49, "VT": 50, "VA": 51, "WA": 53, "WV": 54,
    "WI": 55, "WY": 56
}

# Create data copy and map FIPS IDs
state_cancellations_map = state_cancellations.copy()
state_cancellations_map["id"] = state_cancellations_map["state"].map(state_fips)
state_cancellations_map["cancelled_grants"] = state_cancellations_map["cancelled_grants"].fillna(0)


chart_map2 = (
    alt.Chart(states)
    .mark_geoshape(stroke="white")
    .transform_lookup(
        lookup="id",
        from_=alt.LookupData(
            state_cancellations_map, "id", ["state", "cancelled_grants"]
        ),
    )
    .encode(
        color=alt.Color(
            "cancelled_grants:Q",
            title="Cancelled Grants",
            scale=alt.Scale(
                scheme="blues",
                domain=[1, 500],        
                type="sqrt",            
                interpolate="lab"
            ),
        ),
        tooltip=[
            alt.Tooltip("state:N", title="State"),
            alt.Tooltip("cancelled_grants:Q", title="Cancelled Grants"),
        ],
    )
    .project(type="albersUsa")
    .properties(width=560, height=460)
)


top10_states = state_cancellations.head(10)

chart_bar = (
    alt.Chart(top10_states)
    .mark_bar()
    .encode(
        y=alt.Y("state:N", sort="-x", title=""),
        x=alt.X("cancelled_grants:Q", title="Cancelled Grants"),
        color=alt.Color(
            "cancelled_grants:Q",
            scale=alt.Scale(
                scheme="blues",
                domain=[1, 500],
                type="sqrt",
                interpolate="lab"
            ),
            legend=None
        ),
        tooltip=["state", "cancelled_grants"]
    )
    .properties(width=280, height=460, title="Top 10 States")
)

combined_chart = (chart_map2 | chart_bar).properties(
    title="NSF Grant Cancellations by U.S. State (Smooth Blue Gradient — No White States)"
)

Q1 = combined_chart
#Q1

# Q2
# Q2: Institutions most affected by number of cancelled grants
print("=== Q2: Institutions Most Affected by Number of Cancelled Grants ===")

# Count cancellations by institution
institution_cancellations = terminated_grants['org_name'].value_counts().reset_index()
institution_cancellations.columns = ['institution', 'cancelled_grants']

print(f"Total institutions with cancelled grants: {len(institution_cancellations)}")
print("\nTop 15 institutions by number of cancelled grants:")
print(institution_cancellations.head(15))

# Calculate statistics
print(f"\nStatistics:")
print(f"Mean cancelled grants per institution: {institution_cancellations['cancelled_grants'].mean():.2f}")
print(f"Median cancelled grants per institution: {institution_cancellations['cancelled_grants'].median():.2f}")
print(f"Max cancelled grants by single institution: {institution_cancellations['cancelled_grants'].max()}")

# Create visualization
chart_q2 = alt.Chart(institution_cancellations.head(20)).mark_bar().encode(
    x=alt.X('cancelled_grants:Q', title='Number of Cancelled Grants'),
    y=alt.Y('institution:N', sort='-x', title='Institution'),
    color=alt.Color(
    'cancelled_grants:Q',
    scale=alt.Scale(scheme='blues'),
    legend=alt.Legend(title='Cancelled Grants')
),
    tooltip=['institution', 'cancelled_grants']
).properties(
    width=700,
    height=500,
    title='Top 20 Institutions by Number of Cancelled NSF Grants'
).interactive()

Q2 = chart_q2
#Q2

# Q3

# Q3: Institutions most affected by budget losses
print("=== Q3: Institutions Most Affected by Budget Losses ===")

# Calculate budget impact by institution
budget_impact = terminated_grants.groupby('org_name').agg({
    'nsf_total_budget': ['sum', 'count', 'mean'],
    'nsf_obligated': 'sum',
    'estimated_budget': 'sum'
}).round(2)

# Flatten column names
budget_impact.columns = ['total_budget_sum', 'grant_count', 'avg_budget', 'obligated_sum', 'estimated_sum']
budget_impact = budget_impact.reset_index()

# Use total budget as primary metric, fill with estimated if missing
budget_impact['budget_impact'] = budget_impact['total_budget_sum'].fillna(budget_impact['estimated_sum'])
budget_impact = budget_impact[budget_impact['budget_impact'] > 0].sort_values('budget_impact', ascending=False)

print(f"Institutions with budget data: {len(budget_impact)}")
print(f"\nTop 15 institutions by budget impact (in dollars):")
print(budget_impact[['org_name', 'budget_impact', 'grant_count']].head(15))

# Calculate total budget impact
total_budget_impact = budget_impact['budget_impact'].sum()
print(f"\nTotal budget impact across all institutions: ${total_budget_impact:,.0f}")

# Create visualization
chart_q3 = alt.Chart(budget_impact.head(20)).mark_bar().encode(
    x=alt.X('budget_impact:Q', title='Total Budget Impact ($)', axis=alt.Axis(format='$,.0f')),
    y=alt.Y('org_name:N', sort='-x', title='Institution'),
    color=alt.Color(
    'budget_impact:Q',
    scale=alt.Scale(scheme='blues'),
    legend=alt.Legend(title='Budget Impact ($)', format='$,.0f')
),
    tooltip=[
        alt.Tooltip('org_name', title='Institution'),
        alt.Tooltip('budget_impact:Q', title='Budget Impact', format='$,.0f'),
        alt.Tooltip('grant_count:Q', title='Number of Grants')
    ]
).properties(
    width=700,
    height=500,
    title='Top 20 Institutions by Total Budget Impact from Cancelled NSF Grants'
).interactive()

Q3 = chart_q3
##Q3

# Q4
# Q4 Re-redesgin : Frequency polygon (line) for the left chart
import numpy as np
import pandas as pd
import altair as alt

df_q4 = cleaned_nsf_data.copy()
vals = df_q4["flagged_words_count"].fillna(0).clip(upper=40)

# 1-bin per integer count (0..40)
bins = np.arange(0, 41, 1)
hist, edges = np.histogram(vals, bins=bins)

df_bins = pd.DataFrame({
    "bin_start": edges[:-1],
    "bin_end": edges[1:],
    "count": hist
})
df_bins["mid"] = (df_bins["bin_start"] + df_bins["bin_end"]) / 2  # frequency polygon x

# frequency polygon (points + line through bin midpoints)
chart_q4_hist = (
    alt.Chart(df_bins)
    .mark_line(point=True)
    .encode(
        x=alt.X("mid:Q", title="Number of Flagged Words per Grant"),
        y=alt.Y("count:Q", title="Number of Grants"),
    )
    .properties(width=350, height=250,
                title="Distribution of Flagged Word Counts in Cancelled Grants (Frequency Polygon)")
)
# Q4 (Redesign)

import altair as alt
import pandas as pd
from collections import Counter
import re

print("=== Q4 (Redesign): Distribution + Top Flagged Words in Cancelled Grants ===")

# Base dataset (already filtered to terminated grants)
df_q4 = cleaned_nsf_data.copy()
df_q4["flagged_words_count"] = df_q4["flagged_words_count"].fillna(0)

# Left panel — distribution of how many flagged words appear per cancelled grant
chart_q4_hist = (
    alt.Chart(df_q4)
    .transform_filter(alt.datum.flagged_words_count < 40)
    .mark_bar(opacity=0.8)
    .encode(
        x=alt.X(
            "flagged_words_count:Q",
            bin=alt.Bin(maxbins=40),
            title="Number of Flagged Words per Grant"
        ),
        y=alt.Y("count():Q", title="Number of Grants"),
        color=alt.value("#1d4ed8"),
        tooltip=[
            alt.Tooltip("flagged_words_count:Q", title="Flagged words (binned)"),
            alt.Tooltip("count():Q", title="Grants in bin")
        ]
    )
    .properties(
        width=350,
        height=250,
        title="Distribution of Flagged Word Counts in Cancelled Grants"
    )
)

# 2
flagged_words_list = flagged_words_clean  

# Count flagged word occurrences in title + abstract for terminated grants
all_texts = (
    df_q4.get("project_title", df_q4.get("title", pd.Series("", index=df_q4.index))).fillna("") 
    + " " 
    + df_q4.get("abstract", pd.Series("", index=df_q4.index)).fillna("")
).str.lower()

word_counter = Counter()
for text in all_texts:
    for word in flagged_words_list:
        if re.search(rf"\b{re.escape(word)}\b", text):
            word_counter[word] += 1

# Convert to DataFrame for plotting
df_top_words = (
    pd.DataFrame(word_counter.items(), columns=["word", "count"])
    .sort_values("count", ascending=False)
    .head(15)
)

chart_q4_words = (
    alt.Chart(df_top_words)
    .mark_bar()
    .encode(
        y=alt.Y("word:N", sort="-x", title="Flagged Word"),
        x=alt.X("count:Q", title="Occurrences in Cancelled Grants"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=[
            alt.Tooltip("word:N", title="Word"),
            alt.Tooltip("count:Q", title="Occurrences")
        ]
    )
    .properties(
        width=350,
        height=250,
        title="Top 15 Flagged Words Found in Cancelled Grants"
    )
)


chart_q4_final = chart_q4_hist | chart_q4_words
#chart_q4_final


chart_q4_final = chart_q4_hist | chart_q4_words
##chart_q4_final
##Q4 = chart_q4_final


# Q5
import altair as alt
import pandas as pd

df = cleaned_nsf_data.copy()

# Labels + explicit stack order
df["cruz_label"] = df["in_cruz_list"].map({True: "Yes", False: "No"})
df["status_label"] = df["reinstated"].map({True: "Reinstated", False: "Terminated"})
df["status_order"] = df["status_label"].map({"Terminated": 0, "Reinstated": 1})

# Counts per (Cruz, Status) + row totals and percentages
q5_counts = (
    df.groupby(["cruz_label", "status_label", "status_order"])
    .size()
    .reset_index(name="count")
)
row_totals = (
    q5_counts.groupby("cruz_label")["count"].sum().reset_index(name="row_total")
)
q5_counts = q5_counts.merge(row_totals, on="cruz_label", how="left")
q5_counts["percentage"] = (q5_counts["count"] / q5_counts["row_total"] * 100).round(1)

# Overall totals
totals = (
    df.groupby(["status_label", "status_order"])
    .size()
    .reset_index(name="count")
    .sort_values("status_order")
)
totals["one"] = "Totals"
total_sum = int(totals["count"].sum())
totals["percentage"] = (totals["count"] / total_sum * 100).round(1)

# ---- Enhanced palette ----
BLUE_DARK = "#3182bd"   
BLUE_LIGHT = "#9ecae1"  
GRID = "#E5E7EB"
DARK_TEXT = "#111827"

# ---- Left x-axis ----
x_max = 1600
tick_step = 200
axis_values = list(range(0, x_max + tick_step, tick_step))

# Left panel: stacked bars with percentages

base_left = alt.Chart(q5_counts).properties(width=580, height=250)

stack = base_left.mark_bar().encode(
    y=alt.Y(
        "cruz_label:N",
        title="In Ted Cruz's List",
        sort=["No", "Yes"],
        axis=alt.Axis(labelFontSize=13, titleFontSize=14, titleFontWeight=600),
    ),
    x=alt.X(
        "count:Q",
        title="Number of Grants",
        scale=alt.Scale(domain=[0, x_max], nice=False, zero=True),
        axis=alt.Axis(
            values=axis_values,
            labelExpr='format(datum.value, ",")',
            labelFontSize=12,
            titleFontSize=14,
            titleFontWeight=600,
        ),
    ),
    color=alt.Color(
    "status_label:N",
    title=None,
    scale=alt.Scale(
        domain=["Terminated", "Reinstated"],
        range=[BLUE_DARK, BLUE_LIGHT]
    ),
    legend=None,
),
    order=alt.Order("status_order:Q"),
    tooltip=[
        alt.Tooltip("cruz_label:N", title="In Cruz's List"),
        alt.Tooltip("status_label:N", title="Status"),
        alt.Tooltip("count:Q", title="Count", format=","),
        alt.Tooltip("percentage:Q", title="Percentage", format=".1f"),
    ],
)

# Percentage labels inside bars (centered in each segment)
percentage_labels = (
    base_left.transform_joinaggregate(total="sum(count)", groupby=["cruz_label"])
    .transform_window(
        cum="sum(count)",
        sort=[alt.SortField("status_order", order="ascending")],
        groupby=["cruz_label"],
    )
    .transform_calculate(center="datum.cum - datum.count / 2")
    .mark_text(
        align="center", baseline="middle", fontSize=14, fontWeight=600, color="white"
    )
    .encode(
        y=alt.Y("cruz_label:N", sort=["No", "Yes"]),
        x=alt.X("center:Q"),
        text=alt.Text("percentage:Q", format=".1f"),
        opacity=alt.condition(
            alt.datum.count > 50,  # Only show percentage if segment is large enough
            alt.value(1),
            alt.value(0),
        ),
    )
)

# Row totals at right edge
totals_labels = (
    alt.Chart(row_totals)
    .mark_text(align="left", dx=10, fontSize=13, fontWeight=600, color=DARK_TEXT)
    .encode(
        y=alt.Y("cruz_label:N", sort=["No", "Yes"], title=None),
        x=alt.X("row_total:Q"),
        text=alt.Text("row_total:Q", format=","),
    )
)

left_panel = (stack + percentage_labels + totals_labels).properties(
    title=alt.TitleParams(
        "Grants by Cruz List Status", fontSize=16, fontWeight=600, anchor="start"
    )
)

# Right panel: Totals with centered labels and percentages

right_base = (
    alt.Chart(totals)
    .transform_joinaggregate(total="sum(count)")
    .transform_window(
        cum="sum(count)", sort=[alt.SortField("status_order", order="ascending")]
    )
    .transform_calculate(center="datum.cum - datum.count / 2")
)

totals_bar = (
    right_base.mark_bar()
    .encode(
        x=alt.X("one:N", axis=None, title=""),
        y=alt.Y(
            "count:Q",
            stack="zero",
            axis=None,
            title="",
            scale=alt.Scale(domain=[0, total_sum], nice=False, zero=True),
        ),
        color=alt.Color(
    "status_label:N",
    scale=alt.Scale(
        domain=["Terminated", "Reinstated"],
        range=[BLUE_DARK, BLUE_LIGHT]
    ),
    legend=None,
),
        order=alt.Order("status_order:Q"),
    )
    .properties(
        width=200,
        height=250,
        title=alt.TitleParams(
            "Overall Totals", fontSize=16, fontWeight=600, anchor="middle"
        ),
    )
)

# Status labels with percentages
totals_labels_text = (
    right_base.transform_calculate(
        label='datum.status_label + " (" + toString(datum.percentage) + "%)"'
    )
    .mark_text(baseline="middle", fontSize=14, fontWeight=600, color="white")
    .encode(
        x=alt.X("one:N"),
        y=alt.Y(
            "center:Q",
            axis=None,
            scale=alt.Scale(domain=[0, total_sum], nice=False, zero=True),
        ),
        text=alt.Text("label:N"),
    )
)

right_panel = totals_bar + totals_labels_text

# Combine with enhanced styling

Q5 = (left_panel | right_panel).resolve_scale(color="shared")

#Q5


# --- Mini-panels for dashboard -----------------------------------------------------------------------
F1 = chart_map2.properties(width=260, height=200, title="Q1 – Cancellations by State (Map)")
F2 = chart_bar.properties(width=260, height=200, title="Q1 – Top 10 States by Cancellations")
F3 = chart_q2.properties(width=260, height=200, title="Q2 – Top Institutions by # Cancellations")
F4 = chart_q3.properties(width=260, height=200, title="Q3 – Top Institutions by Budget Loss")
F5 = chart_q4_hist.properties(width=260, height=200, title="Q4 – Flagged Words per Grant")
F6 = chart_q4_words.properties(width=260, height=200, title="Q4 – Top Flagged Words in Cancelled Grants")
F7 = left_panel.properties(width=260, height=200, title="Q5 – Grants by Cruz List Status")
F8 = right_panel.properties(width=260, height=200, title="Q5 – Overall Totals")

final_dashboard = (
    alt.concat(
        F1, F2, F3, F4,
        F5, F6, F7, F8,
        columns=4
    )
    .properties(title="NSF Grant Cancellations — Final Overview (Q1–Q5)")
    .configure_view(strokeWidth=0)
    .configure_axis(
        grid=True,
        gridColor="#E5E7EB",
        gridOpacity=0.5,
        domainColor="#E5E7EB",
        tickColor="#E5E7EB",
        labelColor="#111827",
        titleColor="#111827",
    )
    .configure_concat(spacing=40)
)

final_dashboard = final_dashboard.configure_view(strokeWidth=0).configure_axis(labelColor='white', titleColor='white')


st.title("NSF Grant Cancellations — Final Overview (Q1–Q5)")
st.altair_chart(final_dashboard, use_container_width=True)


