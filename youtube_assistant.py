from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def get_transcript(url: str):
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        transcript = loader.load()
        return transcript
    except Exception as e:
        return f"Error: {str(e)}"


def create_vectorstore(transcript: str):
    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    transcript_chunks = doc_splitter.split_documents(transcript)

    embeddings_model = HuggingFaceEmbeddings()

    return FAISS.from_documents(transcript_chunks, embeddings_model)


def prompt_text():
    prompt = PromptTemplate(
                input_variables=["question", "docs"],
                template="""
                You are a helpful assistant that that can answer questions about youtube videos based on the video's transcript.
                
                Answer the following question: {question}
                by searching the following video transcript: {docs}
                
                Only use the factual information from the transcript to answer the question.
                
                If you feel like you don't have enough information to answer the question, say "I don't know".
                
                Your answers should be verbose and detailed.
                """,
    )
    return prompt


def get_trascript_and_response(url: str, query: str):
    transcript = get_transcript(url)
    
    vectorstore = create_vectorstore(transcript)

    relevant_transcript_chunks = vectorstore.similarity_search(query, k=5)
    relevant_context = " ".join([c.page_content for c in relevant_transcript_chunks])

    prompt = prompt_text()
    inputs = {
        "question": query,
        "docs": relevant_context
    }

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.5,    
    )
    chain = prompt | llm

    return transcript, chain.invoke(inputs)


