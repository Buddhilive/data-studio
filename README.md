# Buddhilive Data Studio

Buddhilive Data Studio is a RAG (Retrieval-Augmented Generation) powered chatbot that allows you to chat with your documents and WordPress data. It uses Streamlit for the user interface, ChromaDB for the vector store, Sentence Transformers for embeddings, and smollm for text generation.

## Features

-   **Document Upload**: Upload and vectorize PDF, DOCX, and TXT files.
-   **WordPress Integration**: Ingest data from WordPress XML exports or SQL dumps.
-   **Chat Interface**: Ask questions and get answers from your data.
-   **Open Source**: Built with open-source models and libraries.

## How to Use

### 1. Environment Setup

It is recommended to use a virtual environment to manage the dependencies for this project.

**Create a virtual environment:**

```bash
python -m venv dojo
```

**Activate the virtual environment:**

-   On Windows:
    ```bash
    dojo\Scripts\activate
    ```
-   On macOS and Linux:
    ```bash
    source dojo/bin/activate
    ```

### 2. Install Dependencies

With your virtual environment activated, install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Note: The `langchain` library has been modularized. This project uses `langchain-community` for vector store access. The `requirements.txt` file includes this dependency.

### 3. Setting up Ollama

This application uses Ollama to run the `smollm` language model locally. You need to have Ollama installed and the `smollm` model pulled before running the application.

1.  **Install Ollama**: Follow the instructions on the [Ollama website](https://ollama.ai/) to download and install it on your system.

2.  **Pull the `smollm` model**: Once Ollama is installed and running, open your terminal and run the following command to download the `smollm` model:
    ```bash
    ollama pull smollm
    ```

3.  **Ensure Ollama is running**: Make sure the Ollama application is running in the background before you start the Streamlit app.

### 4. Running the Application

Once the dependencies are installed, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will open the Buddhilive Data Studio in your web browser.

### 4. Using the App

-   **Data Ingestion**:
    -   Use the sidebar to upload your documents (PDF, DOCX, TXT) and click "Vectorize Documents".
    -   Alternatively, upload a WordPress XML export or SQL dump and click the corresponding "Vectorize" button.
    -   The app will process the files and store the vector embeddings in a local Chroma database (`data/chroma_db`).
-   **Chatting with your Data**:
    -   Once your data has been vectorized, you can start asking questions in the chat interface.
    -   The chatbot will use the information from your documents to provide relevant answers.
