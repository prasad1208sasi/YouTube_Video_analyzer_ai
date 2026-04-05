import os
import zipfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import YoutubeLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch
)
from langchain_core.output_parsers import StrOutputParser

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

# Load env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# -----------------------------
# PROMPTS
# -----------------------------
system_message = "You are a Professional Article Writer."

human_message = """
Transform YouTube transcript into engaging professional article.

IGNORE:
- intro, subscribe, marketing

FOCUS:
- technical insights

FORMAT:
- First-person
- Bold headings
- Code snippets
- Actionable steps
- End summary

{transcript}
"""

summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", human_message)
])

# -----------------------------
# TRANSCRIPT
# -----------------------------
def extract_transcript(link: str) -> str:
    loader = YoutubeLoader.from_youtube_url(link)
    docs = loader.load()
    return docs[0].page_content

# -----------------------------
# BASE SUMMARIZER
# -----------------------------
base_summarizer = (
    RunnablePassthrough()
    | RunnableLambda(extract_transcript)
    | summarizer_prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# TEXT CHUNKING
# -----------------------------
def get_text_chunks(text, chunk_size=5000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

# -----------------------------
# AGENT
# -----------------------------
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=system_message,
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", 1000),
            keep=("tokens", 200),
        )
    ],
)

def recursive_summarize(text):
    chunks = get_text_chunks(text)
    running_summary = ""

    for chunk in chunks:
        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"""
                    Current summary: {running_summary}
                    New content: {chunk}
                    Generate article format.
                    """
                }
            ]
        })

        running_summary = response["messages"][-1].content

    return running_summary

long_summarizer = (
    RunnablePassthrough()
    | RunnableLambda(extract_transcript)
    | RunnableLambda(recursive_summarize)
)

# -----------------------------
# ROUTER
# -----------------------------
def estimate_transcript_length(link: str):
    transcript = extract_transcript(link)
    return len(transcript) > 1000

smart_summarizer = RunnableBranch(
    (RunnableLambda(estimate_transcript_length), long_summarizer),
    base_summarizer
)

# -----------------------------
# WEBPAGE GENERATOR
# -----------------------------
web_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a frontend developer.

Return ONLY:
--html-- code --html--
--css-- code --css--
--js-- code --js--
"""),
    ("human", "Create article webpage for: {article}")
])

web_chain = web_prompt | llm | StrOutputParser()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="YouTube → Article Generator", layout="wide")

st.title("🎬 YouTube → Article + Website Generator")

url = st.text_input("Enter YouTube URL")

if st.button("Generate 🚀"):
    if url:
        with st.spinner("Processing... ⏳"):

            article = smart_summarizer.invoke(url)

            webpage = web_chain.invoke({"article": article})

            # Split output
            html = webpage.split("--html--")[1]
            css = webpage.split("--css--")[1]
            js = webpage.split("--js--")[1]

            # Save files
            with open("index.html", "w") as f:
                f.write(html)

            with open("style.css", "w") as f:
                f.write(css)

            with open("script.js", "w") as f:
                f.write(js)

            # Zip
            zip_path = "website.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                z.write("index.html")
                z.write("style.css")
                z.write("script.js")

            st.success("✅ Done!")

            st.subheader("📄 Generated Article")
            st.write(article)

            with open(zip_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Website",
                    f,
                    file_name="website.zip"
                )