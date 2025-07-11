import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Streamlit page config
st.set_page_config(page_title="Langchain: Summarize Text From YouTube or Website")
st.title("Langchain: Summarize Text From YouTube or Website")
st.subheader('Summarize URL')

# URL input
generic_url = st.text_input("URL", label_visibility="collapsed")

# LLM setup
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Prompt template
promptTemplate = """
If you see a code just explain the code briefly.
Provide the answer in a well-structured way.
The answer should be in bullet form.
Provide a summary of the following content:
Content: {text}
"""
prompt = PromptTemplate(template=promptTemplate, input_variables=["text"])

# Function to extract YouTube transcript
def get_youtube_transcript(video_url):
    parsed_url = urlparse(video_url)
    if parsed_url.hostname == 'youtu.be':
        video_id = parsed_url.path[1:]
    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            video_id = parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith('/embed/'):
            video_id = parsed_url.path.split('/')[2]
        elif parsed_url.path.startswith('/v/'):
            video_id = parsed_url.path.split('/')[2]
        else:
            video_id = None
    else:
        video_id = None

    if not video_id:
        raise ValueError("Invalid YouTube URL format.")

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([i["text"] for i in transcript])
    return text

# Main action button
if st.button("Summarize the content from YouTube or website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video or website URL.")
    else:
        try:
            with st.spinner("Processing..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        text = get_youtube_transcript(generic_url)
                        docs = [Document(page_content=text)]
                    except Exception:
                        st.error("The transcript of this YouTube video is private or not available.")
                        st.stop()
                else:
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
                            },
                        )
                        docs = loader.load()
                    except Exception:
                        st.error("The content of this website is private or cannot be accessed without permission.")
                        st.stop()

                # Summarization
                chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
