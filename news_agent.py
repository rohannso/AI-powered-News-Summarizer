import os
import streamlit as st
import langgraph.graph as lg
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from pydantic import BaseModel
from typing import List
import pyttsx3

# -----------------------------
# 🔑 Set up Groq API Key
# -----------------------------
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key
llm = ChatGroq(model_name="mixtral-8x7b-32768")

# -----------------------------
# 🎭 Define the State Schema
# -----------------------------
class NewsState(BaseModel):
    query: str
    articles: List[dict] = []
    summaries: List[dict] = []
    categorized_news: List[dict] = []

# -----------------------------
# 📰 Function to Fetch News
# -----------------------------
def search_news(state: NewsState) -> dict:
    """Fetch news articles using DuckDuckGo search."""
    with DDGS() as ddgs:
        results = list(ddgs.news(state.query, max_results=5))
    return {"articles": results}

# -----------------------------
# ✍️ Function to Summarize News
# -----------------------------
def summarize_news(state: NewsState) -> dict:
    """Summarize fetched news using LLM."""
    summaries = []
    for article in state.articles:
        prompt = f"Summarize this news article: {article['title']} - {article.get('body', '')}"
        response = llm.invoke(prompt)
        summary_text = response.content  # Extract only the text

        summaries.append({"title": article['title'], "summary": summary_text})
    return {"summaries": summaries}

# -----------------------------
# 📌 Function to Categorize News
# -----------------------------
def categorize_news(state: NewsState) -> dict:
    """Categorize news into Politics, Tech, Sports, Business, or Others."""
    categorized_news = []
    for item in state.summaries:
        prompt = f"Classify this news article into a category (Politics, Tech, Sports, Business, Others): {item['summary']}"
        category = llm.invoke(prompt)
        category_text = category.content  # Extract only the text

        categorized_news.append({"title": item['title'], "summary": item['summary'], "category": category_text})
    return {"categorized_news": categorized_news}

# -----------------------------
# 🔗 Define the LangGraph Workflow with State
# -----------------------------
workflow = lg.StateGraph(NewsState)  # Define State Schema
workflow.add_node("search_news", search_news)
workflow.add_node("summarize_news", summarize_news)
workflow.add_node("categorize_news", categorize_news)

# Define workflow edges (flow)
workflow.add_edge("search_news", "summarize_news")
workflow.add_edge("summarize_news", "categorize_news")

# Set entry and exit points
workflow.set_entry_point("search_news")
workflow.set_finish_point("categorize_news")

# Compile the graph
news_graph = workflow.compile()

# -----------------------------
# 🎙️ Text-to-Speech Function
# -----------------------------
engine = pyttsx3.init()

def speak_text(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

# Optional: Customize voice settings
engine.setProperty('rate', 150)  # Speed of speech (default: 200)
engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 0: Male, 1: Female (Change as needed)

# -----------------------------
# 🌟 Streamlit UI
# -----------------------------
st.title("📰 AI-Powered News Summarizer & Categorizer")
st.write("🔍 Enter a topic to get summarized and categorized news!")

# Input field
topic = st.text_input("Enter News Topic", placeholder="e.g., AI advancements, SpaceX, World Politics")

# Run the workflow when button is clicked
if st.button("Fetch & Summarize News"):
    if topic:
        with st.spinner("Fetching and processing news... ⏳"):
            initial_state = NewsState(query=topic)
            output_dict = news_graph.invoke(initial_state)  # ✅ Get dictionary output

        st.subheader("🔹 Categorized News")
        for item in output_dict.get("categorized_news", []):  # ✅ Use .get() for safety
            st.markdown(f"### 📰 {item['title']}")
            st.write(f"**Category:** {item['category']}")
            st.write(f"**Summary:** {item['summary']}")
            st.divider()
            
            # Speak the summary
            speak_text(f"{item['title']}. Category: {item['category']}. Summary: {item['summary']}")

    else:
        st.warning("⚠️ Please enter a topic to search!")
