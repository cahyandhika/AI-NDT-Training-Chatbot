import streamlit as st
import os
import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize LLM model
model = "deepseek-r1-distill-llama-70b"
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

# Define prompt template
PROMPT_TEMPLATE = """Human: You are an AI assistant, and provide answers using fact-based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question>.
If you don't know the answer, just say that you don't know, don't try to make up an answer. just answer with the same language as the question.
<context>{context}</context>
<question>{question}</question>
Assistant:"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# Initialize embeddings and Pinecone retriever
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("ndt-guidebot")
vectorstore = PineconeVectorStore(index=index, embedding=embedding)
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | groq_chat
    | StrOutputParser()
)

# Initialize conversation history with timestamps
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("AI NDT Training Chatbot: Insights from BINDT & IAEA Publications")

# Sidebar Conversation History with Collapsible Section
with st.sidebar:
    st.header("Conversation History")
    with st.expander("View Chat History"):
        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                st.write(f"**{entry['timestamp']}** - {entry['message']}")
        else:
            st.write("No conversation history yet.")

    # Clear history button
    if st.button("Clear History"):
        st.session_state.chat_history = []

# Main chat interface
with st.form("my_form"):
    text = st.text_area("Ask a question about NDT:", "What is NDT?")
    submitted = st.form_submit_button("Submit")

    if submitted:
        response = rag_chain.invoke(text)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({"timestamp": timestamp, "message": f"You: {text}"})
        st.session_state.chat_history.append({"timestamp": timestamp, "message": f"Bot: {response}"})
        st.write(response)

