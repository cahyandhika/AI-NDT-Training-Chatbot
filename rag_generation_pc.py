import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Groq API key
groq_api_key = os.environ["GROQ_API_KEY"]
model = 'deepseek-r1-distill-llama-70b'

#initialize Groq Langchain chat object and conversation

groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model,
)

# Define the prompt template for generating AI responses
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <query>,
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possibe.
If otherwise specified, please answer always with the same language as the question.

Assistant:"""

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# prompt_chain = prompt | groq_chat | StrOutputParser() | RunnablePassthrough()
# result = prompt_chain.invoke({"context": "", "question": "Siapa penulis dari module GenAI & Langchain workshop?"})
# print(result)

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

from pinecone import Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

from langchain_pinecone import PineconeVectorStore
index_name = "bindt-384"

# Instantiate the Pinecone index
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(index=index, embedding=embedding)
retriever = vectorstore.as_retriever()

# Define a function to format the retrieved documents for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | groq_chat
    | StrOutputParser()
)

response = rag_chain.invoke("from all of the ndt method, which one is the most suitable for detecting defects in a weld?")
print(response)

