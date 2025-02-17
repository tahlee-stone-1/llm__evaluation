import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher

# Load data
df = pd.read_csv("example_data.csv")

# Streamlit app title
st.title("LLM Comparator - Chatbot Response")

# Sidebar Filters
st.sidebar.header("Filters")
accuracy_filter = st.sidebar.multiselect("Filter by Accuracy", df["accuracy_1"].unique())
relevancy_filter = st.sidebar.multiselect("Filter by Relevancy", df["relevancy_1"].unique())
tone_filter = st.sidebar.multiselect("Filter by Tone", df["tone_1"].unique())
groundedness_filter = st.sidebar.multiselect("Filter by Groundedness", df["groundedness_1"].unique())
l1_cluster_category_filter = st.sidebar.multiselect("Filter by Cluster", df["l1_cluster_category"].unique())

# Filtering Data
filtered_df = df.copy()
if accuracy_filter:
    filtered_df = filtered_df[filtered_df["accuracy_1"].isin(accuracy_filter)]
if relevancy_filter:
    filtered_df = filtered_df[filtered_df["relevancy_1"].isin(relevancy_filter)]
if tone_filter:
    filtered_df = filtered_df[filtered_df["tone_1"].isin(tone_filter)]
if groundedness_filter:
    filtered_df = filtered_df[filtered_df["groundedness_1"].isin(groundedness_filter)]
if l1_cluster_category_filter:
    filtered_df = filtered_df[filtered_df["l1_cluster_category"].isin(l1_cluster_category_filter)]

# Grouped Bar Charts for Comparisons
st.subheader("Response Comparisons")

metrics = ["accuracy", "relevancy", "groundedness", "tone"]
figs = []
col1, col2, col3 = st.columns(3)
for i, metric in enumerate(metrics):
    category_counts = filtered_df.groupby([metric+"_1", metric+"_2"]).size().reset_index(name="count")
    fig = px.bar(category_counts, x=metric+"_1", y="count", color=metric+"_2", barmode='group', title=f"{metric.capitalize()} Comparison")
    fig.update_layout(barmode='group', title=f"{metric.capitalize()} Comparison")
    if i == 0:
        col1.plotly_chart(fig, use_container_width=True)
    elif i == 1:
        col2.plotly_chart(fig, use_container_width=True)
    elif i == 2:
        col3.plotly_chart(fig, use_container_width=True)

# Low Relevancy Rate by Cluster
st.subheader("Low Relevancy Rate by Cluster")
low_relevancy_df = filtered_df[filtered_df["relevancy_1"] == "Low"]
low_relevancy_counts = low_relevancy_df.groupby(["l1_cluster_category", "relevancy_2"]).size().reset_index(name="count")
fig = px.bar(low_relevancy_counts, x="l1_cluster_category", y="count", color="relevancy_2", barmode='group', title="Low Relevancy Counts by Cluster and Relevancy 2")
st.plotly_chart(fig)

# Highlighting Matching Text in Green in Data Table
def highlight_common_text(row):
    text_1, text_2 = row["response_1"], row["response_2"]
    matcher = SequenceMatcher(None, text_1, text_2)
    highlighted_text = ""
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            highlighted_text += f"<span style='color:green'>{text_1[i1:i2]}</span>"
        else:
            highlighted_text += text_1[i1:i2]
    
    return highlighted_text

filtered_df["highlighted_response_1"] = filtered_df.apply(highlight_common_text, axis=1)

st.subheader("Chatbot Response Comparison")
st.write(filtered_df[["query", "highlighted_response_1", "response_2", "accuracy_1", "accuracy_2", "tone_1", "tone_2"]].to_html(escape=False), unsafe_allow_html=True)

# Calculate and Visualize Change in Similarity Score
st.subheader("Change in Similarity Score Between Model Runs")
df["similarity_change"] = df["prompt_answer_relevancy_2"].astype(float) - df["prompt_answer_relevancy_1"].astype(float)
fig = px.histogram(df, x="similarity_change", title="Change in Similarity Score Distribution")
st.plotly_chart(fig)
