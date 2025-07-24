import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

# --- App Configuration ---
st.set_page_config(page_title="Buddhilive Data Studio", layout="wide")
st.title("Buddhilive Data Studio")

# --- Global Variables ---
DATA_DIR = "data"
DB_DIR = os.path.join(DATA_DIR, "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

# --- Configuration ---
def get_api_key():
    """Get API key from various sources"""
    # Try session state first
    if "google_api_key" in st.session_state and st.session_state.google_api_key:
        return st.session_state.google_api_key
    
    # Try Streamlit secrets (with proper error handling)
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        # Secrets file doesn't exist or key not found, continue to next option
        pass
    
    # Try environment variable
    return os.getenv("GOOGLE_API_KEY")

# --- Core Functions ---

@st.cache_resource
def get_embeddings():
    """Initializes and returns the sentence transformer embeddings model."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    """Initializes and returns the Gemini language model."""
    try:
        # Get API key
        api_key = get_api_key()
        if not api_key:
            st.error("Google API key not found. Please enter your API key in the sidebar.")
            st.info("To get an API key, visit: https://makersuite.google.com/app/apikey")
            return None
        
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            max_tokens=512,
            timeout=60,
            max_retries=2,
            google_api_key=api_key
        )
        
        st.success("Gemini API connected successfully!")
        return llm
        
    except Exception as e:
        st.error(f"Error connecting to Gemini API: {str(e)}")
        st.info("Please check your API key and internet connection.")
        return None

def process_documents(uploaded_files):
    """
    Processes uploaded documents, splits them into chunks, and creates a vector store.
    """
    if not uploaded_files:
        st.warning("Please upload at least one document.")
        return None

    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=DB_DIR)
    vectorstore.persist()
    return vectorstore

def process_wordpress_xml(xml_file):
    """
    Parses a WordPress XML export, extracts content, cleans HTML, 
    and creates a vector store.
    """
    st.info(f"Parsing XML file: {xml_file.name}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    content_list = []
    # Namespace dictionary to handle the XML namespaces
    ns = {
        'content': 'http://purl.org/rss/1.0/modules/content/',
        'wp': 'http://wordpress.org/export/1.2/',
    }

    for item in root.findall('channel/item'):
        # We are interested in posts and pages
        post_type = item.find('wp:post_type', ns).text
        if post_type in ['post', 'page']:
            title = item.find('title').text
            encoded_content = item.find('content:encoded', ns).text
            
            if encoded_content:
                # Use BeautifulSoup to strip HTML tags
                soup = BeautifulSoup(encoded_content, 'lxml')
                text_content = soup.get_text()
                # Combine title and content for better context
                full_content = f"Title: {title}\n\n{text_content}"
                content_list.append(full_content)

    if not content_list:
        st.warning("No posts or pages found in the XML file.")
        return None

    # Convert the list of strings to LangChain Document objects
    from langchain.schema import Document
    documents = [Document(page_content=text) for text in content_list]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=DB_DIR)
    vectorstore.persist()
    st.success(f"Successfully processed and vectorized {len(content_list)} items from XML.")
    return vectorstore

def process_wordpress_sql(sql_file):
    """
    Parses a WordPress SQL dump, extracts content from the posts table, 
    cleans it, and creates a vector store.
    Note: This is a simplified parser and may not handle all SQL dump variations.
    """
    st.info(f"Parsing SQL file: {sql_file.name}")
    
    # Read the content of the uploaded file
    sql_content = sql_file.getvalue().decode("utf-8")
    
    content_list = []
    
    # A simple regex to find post content within INSERT statements for wp_posts
    # This is a basic approach and might need refinement for different SQL dump formats
    import re
    
    # This regex looks for the post_content column in an INSERT statement
    # It's tricky because the content can contain escaped quotes and newlines.
    # We'll look for the full INSERT statements and then parse them.
    insert_pattern = re.compile(r"INSERT INTO `wp_posts` VALUES\s*\((.*?)\);", re.DOTALL)
    
    for match in insert_pattern.finditer(sql_content):
        values_str = match.group(1)
        # This is a very fragile way to parse SQL values. A proper SQL parser would be better.
        # We'll split by comma, but this can fail if commas are in the content.
        # A more robust way is to find the post_content by its position.
        # Assuming standard wp_posts structure, post_content is the 5th column.
        try:
            # Splitting the values string is complex due to escaped quotes.
            # Let's try a more direct approach by finding the content between quotes.
            # This is still not perfect.
            # A better way is to find the start of the post_content field.
            # Let's assume we can find it by looking for the post_title, which is usually before it.
            
            # A simplified parsing logic:
            # We split the values string by comma and hope for the best.
            values = values_str.split(',')
            if len(values) > 5:
                # The post_content is likely the 5th value (index 4)
                # It will be enclosed in single quotes.
                post_content_raw = values[4].strip()
                if post_content_raw.startswith("'") and post_content_raw.endswith("'"):
                    post_content_escaped = post_content_raw[1:-1]
                    # Unescape SQL quotes
                    post_content = post_content_escaped.replace("\\'", "'").replace('\\"', '"')
                    
                    # Clean HTML
                    soup = BeautifulSoup(post_content, 'lxml')
                    text_content = soup.get_text()
                    
                    if text_content and len(text_content.strip()) > 50: # Basic filter for meaningful content
                        content_list.append(text_content)

        except Exception as e:
            st.warning(f"Could not parse a row: {e}")
            continue

    if not content_list:
        st.warning("No post content found in the SQL file. The parser might need adjustments for your specific SQL dump format.")
        return None

    from langchain.schema import Document
    documents = [Document(page_content=text) for text in content_list]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=DB_DIR)
    vectorstore.persist()
    st.success(f"Successfully processed and vectorized {len(content_list)} items from SQL.")
    return vectorstore


# --- UI Layout ---

st.sidebar.title("Configuration")

# API Key Input
current_api_key = get_api_key()
with st.sidebar.expander("🔑 API Configuration", expanded=not current_api_key):
    st.markdown("**Google Gemini API Key**")
    api_key_input = st.text_input(
        "Enter your Google API Key:",
        type="password",
        value=current_api_key if current_api_key else "",
        help="Get your API key from https://makersuite.google.com/app/apikey",
        key="api_key_input"
    )
    
    if api_key_input:
        # Store in session state
        st.session_state.google_api_key = api_key_input
        if api_key_input != current_api_key:
            st.cache_resource.clear()  # Clear cached LLM to reload with new key
            st.rerun()
    
    if not current_api_key:
        st.warning("⚠️ Please enter your Google API key to use the chatbot.")
        st.markdown("[Get your free API key here](https://makersuite.google.com/app/apikey)")

st.sidebar.title("Data Ingestion")

# Document Upload
st.sidebar.header("Upload Documents")
doc_files = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT files", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
if st.sidebar.button("Vectorize Documents"):
    with st.spinner("Vectorizing documents..."):
        process_documents(doc_files)
        st.sidebar.success("Documents vectorized successfully!")

# WordPress XML Upload
st.sidebar.header("Import from WordPress XML")
xml_file = st.sidebar.file_uploader("Upload WordPress XML Export", type=['xml'])
if st.sidebar.button("Vectorize from XML"):
    if xml_file:
        with st.spinner("Processing XML and vectorizing..."):
            process_wordpress_xml(xml_file)
            st.sidebar.success("Data from XML vectorized successfully!")
    else:
        st.sidebar.warning("Please upload an XML file.")

# WordPress SQL Upload
st.sidebar.header("Import from WordPress SQL Dump")
sql_file = st.sidebar.file_uploader("Upload WordPress SQL Dump", type=['sql'])
if st.sidebar.button("Vectorize from SQL"):
    if sql_file:
        with st.spinner("Processing SQL and vectorizing..."):
            process_wordpress_sql(sql_file)
            st.sidebar.success("Data from SQL vectorized successfully!")
    else:
        st.sidebar.warning("Please upload a SQL file.")


# --- Chat Interface ---

st.header("RAG Powered Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                embeddings = get_embeddings()
                vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                retriever = vectorstore.as_retriever()
                
                llm = get_llm()
                if llm is None:
                    response = "Sorry, I'm unable to load the language model at the moment. Please try again later."
                else:
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                    response = qa_chain.run(prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
