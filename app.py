import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("example_data.csv")

# Streamlit app title
st.title("LLM Comparator - Chatbot Response Analysis")

# Sidebar Filters
st.sidebar.header("Filters")
accuracy_filter = st.sidebar.multiselect("Filter by Accuracy", df["accuracy_1"].unique())
relevancy_filter = st.sidebar.multiselect("Filter by Relevancy", df["relevancy_1"].unique())
tone_filter = st.sidebar.multiselect("Filter by Tone", df["tone_1"].unique())

# Filtering Data
filtered_df = df.copy()
if accuracy_filter:
    filtered_df = filtered_df[filtered_df["accuracy_1"].isin(accuracy_filter)]
if relevancy_filter:
    filtered_df = filtered_df[filtered_df["relevancy_1"].isin(relevancy_filter)]
if tone_filter:
    filtered_df = filtered_df[filtered_df["tone_1"].isin(tone_filter)]

# Display Data Table
st.subheader("Chatbot Response Comparison")
st.dataframe(filtered_df[["query", "response_1", "response_2", "accuracy_1", "accuracy_2", "tone_1", "tone_2"]])

# Semantic Similarity Scatter Plot
st.subheader("Semantic Similarity Distribution")
fig = px.scatter(
    filtered_df, x="prompt_answer_relevancy_1", y="prompt_answer_relevancy_2", 
    color="accuracy_1", hover_data=["query", "response_1", "response_2"],
    title="Semantic Similarity Scores"
)
st.plotly_chart(fig)

# Run using: streamlit run script.py
