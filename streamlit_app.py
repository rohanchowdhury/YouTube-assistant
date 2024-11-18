import streamlit as st
from youtube_assistant import get_trascript_and_response


def is_valid_youtube_url(url):    
    return url.startswith(('https://www.youtube.com/', 'https://youtu.be/'))


st.title("YouTube Assistant")
url = st.text_input("Enter YouTube Video URL:")

if url:
    if is_valid_youtube_url(url):
        query = st.text_input("Enter your query:")
        if query:
            with st.spinner("Generating response..."):

                transcript, ai_response = get_trascript_and_response(url, query)

                if isinstance(transcript, list) and len(transcript) > 0:                
                    st.subheader("AI response:")
                    st.text(ai_response)    
                    
                    if st.button("Press to view the entire transcript!"):
                        st.subheader("Video Transcript:")
                        st.text(transcript[0].page_content)                            
                else:
                    st.error("Failed to generate transcript. Please try again.")
    else:
        st.error("Please enter a valid YouTube URL.")




