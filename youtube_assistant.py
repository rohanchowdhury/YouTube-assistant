from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

def get_transcript(url: str)  -> list[Document]:
    """
    Extracts the transcript from a YouTube video given its URL.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        list or str: A list of transcript segments if successful, or an error message (str) if an exception occurs.

    Raises:
        Exception: If the transcript cannot be retrieved due to an issue with the URL or loading process.
    """
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        transcript = loader.load()
        return transcript
    except Exception as e:
        return f"Error: {str(e)}"


def create_vectorstore(transcript: list[Document]) -> FAISS:
    """
    Creates a FAISS vector store from a YouTube video transcript for efficient semantic search and retrieval.

    Args:
        transcript (str): The full text or content of the YouTube video transcript.

    Returns:
        FAISS: A FAISS vector store containing embeddings for the transcript chunks, ready for retrieval operations.

    Process:
        - Splits the transcript into overlapping chunks for efficient embedding generation and search.
        - Uses the HuggingFaceEmbeddings model to generate embeddings for each chunk.
        - Creates a FAISS vector store from the chunk embeddings.

    Raises:
        ValueError: If the transcript is empty or invalid.
    """
    if not isinstance(transcript, list) or not isinstance(transcript[0], Document):
        raise ValueError("The transcript must be a list of langchain_core.documents.Document")

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
    """
    Extracts the transcript of a YouTube video and generates a LLM response 
    based on the transcript content relevant to the user's query.

    Args:
        url (str): The URL of the YouTube video.
        query (str): The question or query to ask based on the video's transcript.

    Returns:
        tuple:
            - transcript (str): The full transcript extracted from the YouTube video.
            - response (str): The LLM-generated response to the query, derived from the 
              most relevant chunks of the transcript.

    Process:
        1. **Extract Transcript**: The `get_transcript` function retrieves the video transcript.
        2. **Create Vector Store**: The `create_vectorstore` function splits the transcript into chunks, 
           embeds them using a pre-trained model, and stores the embeddings in a FAISS vector store.
        3. **Find Relevant Context**:
            - A similarity search retrieves the top 5 most relevant transcript chunks 
              based on the user's query.
            - The content of these chunks is joined into a single context string.
        4. **Generate Response**:
            - Constructs a prompt using `prompt_text()` to structure the input for the LLM.
            - Uses a HuggingFace hosted LLM endpoint (`Mistral-7B-Instruct-v0.2`) to generate a response.
            - The query and relevant context are passed as inputs to the LLM.

    Note:
        - The function depends on several helper methods: `get_transcript`, `create_vectorstore`, and `prompt_text`.
        - Ensure the necessary packages for handling HuggingFace models and FAISS are installed and api_keys configured.

    """
     
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
