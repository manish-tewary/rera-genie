# pip install tiktoken langchain openai streamlit pypdf2 python-dotenv faiss-cpu huggingface_hub InstructorEmbedding sentence_transformers
# https://www.youtube.com/watch?v=dXxQ0LR-3Hg
# knowledge-hub
# https://www.dataquest.io/blog/read-file-python/ - to read data from csv
# cd talkgpt
# source myVenv/bin/activate
# streamlit run streamlit_app.py

# Standard library import
import logging
import pathlib
import streamlit as st
from io import StringIO
from dotenv import load_dotenv
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_content(docs):
    files = {}
    for file in docs:
        filename, file_extension = os.path.splitext(file.name)
        if file_extension == '.txt':
            stringio = StringIO(file.getvalue().decode("utf-8"))
            text = stringio.read()
            files[filename] = text
        elif file_extension == '.pdf':
            text = ""
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
            files[filename] = text
    return files


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory)
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def process_all_text(docs):
    # texts = get_content(docs)
    pickle_jar = f"{pathlib.Path(__file__).parent.resolve()}/pickle_jar"
    logger.info(f"pickle Jar: %s" % pickle_jar)
    if not os.path.exists(f"{pickle_jar}"):
        os.makedirs(pickle_jar)
    texts = get_content(docs)
    for filename, content in texts.items():
        text_chunks = get_text_chunks(content)
        vector_store = get_vector_store(text_chunks)

        with open(f"{pickle_jar}/{filename}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
    return vector_store


def main():
    load_dotenv()
    st.set_page_config(page_title="ReraGenie", page_icon=":genie:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ReraGenie :genie:")
    user_query = st.text_input("Ask your query:")
    logger.info(f"Message sent from user: {user_query}")
    if user_query:
        handle_user_input(user_query)

    # st.write(user_template.replace("{{MSG}}", "Hello Bot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents:")
        pdf_docs = st.file_uploader(
            "Upload your files here and click on 'Process'", type=['pdf', 'txt', 'csv', 'xlsx'],
            accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                vector_store = process_all_text(pdf_docs)
                # create a conversation chain with memory
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()
