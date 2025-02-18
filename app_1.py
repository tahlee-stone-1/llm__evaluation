import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher

# Load data
df = pd.read_csv("example_data.csv")

# Streamlit app title
st.set_page_config(layout="wide")
st.title("LLM Comparator - Chatbot Responses")

# Top Panel Metrics

col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
filtered_df = df.copy()

num_observations = len(filtered_df)
failure_rate_1 = (filtered_df["relevancy_1"] == "Low").mean() * 100
failure_rate_2 = (filtered_df["relevancy_2"] == "Low").mean() * 100
ungroundedness_rate_1 = (filtered_df["groundedness_1"] == "Low").mean() * 100
ungroundedness_rate_2 = (filtered_df["groundedness_2"] == "Low").mean() * 100
coverage_rate_1 = (filtered_df["non_response_1"] == "No").mean() * 100
coverage_rate_2 = (filtered_df["non_response_2"] == "No").mean() * 100

# Display metrics
st.markdown("### 📊 Model Run Metrics")
st.markdown("---")

# Ensure this appears before metrics
st.markdown("<style>div[data-testid='stMetric'] {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}</style>", unsafe_allow_html=True)
    
col1.metric("# Observations", num_observations)
col2.metric("Failure Rate (Low Relevancy)", f"{failure_rate_1:.1f}% | {failure_rate_2:.1f}%", help="Model 1 | Model 2")

col3.metric("Ungroundedness Rate (Low Groundedness)", f"{ungroundedness_rate_1:.1f}% | {ungroundedness_rate_2:.1f}%", help="Model 1 | Model 2")

col4.metric("Coverage Rate (Non-Response = No)", f"{coverage_rate_1:.1f}% | {coverage_rate_2:.1f}%", help="Model 1 | Model 2")


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
col1, col2, col3 = st.columns([1, 1, 1])
for i, metric in enumerate(metrics):
    category_counts = filtered_df.melt(id_vars=["query"], value_vars=[metric+"_1", metric+"_2"], var_name="Model Run", value_name=metric).groupby([metric, "Model Run"]).size().reset_index(name="count")
    
    fig = px.bar(category_counts, x=metric, y='count', color='Model Run', barmode='group', title=f"{metric.capitalize()} Comparison", color_discrete_map={metric+'_1': 'darkblue', metric+'_2': 'lightblue'})
    fig.update_layout(barmode='group', title=f"{metric.capitalize()} Comparison", height=400, width=400)
    if i == 0:
        col1.plotly_chart(fig, use_container_width=True)
    elif i == 1:
        col2.plotly_chart(fig, use_container_width=True)
    elif i == 2:
        col3.plotly_chart(fig, use_container_width=True)

# Low Relevancy Rate and Coverage Rate by Cluster
col1, col2 = st.columns(2)
col1.subheader("Low Relevancy Rate by Cluster")
low_relevancy_df = filtered_df[filtered_df["relevancy_1"] == "Low"]
low_relevancy_counts = filtered_df.groupby("l1_cluster_category").agg(low_relevancy_1=("relevancy_1", lambda x: (x == "Low").mean()), low_relevancy_2=("relevancy_2", lambda x: (x == "Low").mean())).reset_index()
fig = px.bar(low_relevancy_counts, y='l1_cluster_category', x=['low_relevancy_1', 'low_relevancy_2'], barmode='group', title='Low Relevancy Rate by Cluster', labels={'value': 'Low Relevancy Rate', 'variable': 'Model Run'}, color_discrete_map={'low_relevancy_1': 'darkblue', 'low_relevancy_2': 'lightblue'})

# Coverage Rate by Cluster
coverage_counts = filtered_df.groupby("l1_cluster_category").agg(coverage_1=("non_response_1", lambda x: (x == "No").mean()), coverage_2=("non_response_2", lambda x: (x == "No").mean())).reset_index()
fig_coverage = px.bar(coverage_counts, y='l1_cluster_category', x=['coverage_1', 'coverage_2'], barmode='group', title='Coverage Rate by Cluster', labels={'value': 'Coverage Rate', 'variable': 'Model Run'}, color_discrete_map={'coverage_1': 'darkblue', 'coverage_2': 'lightblue'})
col2.subheader("Coverage Rate by Cluster")
col2.plotly_chart(fig_coverage, use_container_width=True)
col1.plotly_chart(fig, use_container_width=True)

# Highlighting Matching Text in Green in Data Table
import re

def highlight_common_text(row):
    text_1, text_2 = row["response_1"], row["response_2"]
    words_1 = re.findall(r'\b\w+\b', text_1)
    words_2 = re.findall(r'\b\w+\b', text_2)
    
    common_phrases = set(words_1) & set(words_2)
    
    for phrase in sorted(common_phrases, key=len, reverse=True):
        text_1 = re.sub(fr'\b{re.escape(phrase)}\b', f"<span style='background-color:lightgreen'>{phrase}</span>", text_1)
        text_2 = re.sub(fr'\b{re.escape(phrase)}\b', f"<span style='background-color:lightgreen'>{phrase}</span>", text_2)
    
    return text_1, text_2

filtered_df[["response_1", "response_2"]] = filtered_df.apply(lambda row: pd.Series(highlight_common_text(row)), axis=1)

st.subheader("Chatbot Response Comparison")

# Toggle between modes
mode = st.radio("Select Table Mode:", ["Highlighted Mode", "Filterable Mode"])

if mode == "Highlighted Mode":
    # Display table with HTML rendering for highlighted text



    st.write(filtered_df[["l1_cluster_category", "query", "response_1", "response_2", "groundedness_1", "groundedness_2", "relevancy_1", "relevancy_2", "accuracy_1", "accuracy_2"] + [col for col in filtered_df.columns if col not in ["l1_cluster_category", "query", "response_1", "response_2", "groundedness_1", "groundedness_2", "relevancy_1", "relevancy_2", "accuracy_1", "accuracy_2"]]].to_html(escape=False), unsafe_allow_html=True)

elif mode == "Filterable Mode":
    st.data_editor(filtered_df, use_container_width=True, height=600)



# Calculate and Visualize Change in Similarity Score
st.subheader("Change in Similarity Score Between Model Runs")
df["relevancy_change"] = df["prompt_answer_relevancy_2"].astype(float) - df["prompt_answer_relevancy_1"].astype(float)
fig = px.histogram(df, x="similarity_change", title="Change in Similarity Score Distribution")
st.plotly_chart(fig)
