import os
import streamlit as st
import langgraph.graph as lg
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from pydantic import BaseModel
from typing import List

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
        for idx, item in enumerate(output_dict.get("categorized_news", [])):  # ✅ Use .get() for safety
            st.markdown(f"### 📰 {item['title']}")
            st.write(f"**Category:** {item['category']}")
            st.write(f"**Summary:** {item['summary']}")
            st.divider()

            # Unique identifier for each news item
            unique_id = f"text_{idx}"

            # JavaScript Speech Synthesis via iframe embedding
            js_code = f"""
            <script>
                function speakText_{unique_id}() {{
                    var msg = new SpeechSynthesisUtterance("{item['title']} - {item['summary']}");
                    window.speechSynthesis.cancel();  // Stop previous speech
                    window.speechSynthesis.speak(msg);
                }}

                function stopSpeech_{unique_id}() {{
                    window.speechSynthesis.cancel();
                }}
            </script>

            <button onclick="speakText_{unique_id}()">🔊 Speak</button>
            <button onclick="stopSpeech_{unique_id}()" style="margin-left: 10px;">⏹️ Stop</button>
            """

            # Embed JavaScript buttons
            st.components.v1.html(js_code, height=60)

    else:
        st.warning("⚠️ Please enter a topic to search!")
