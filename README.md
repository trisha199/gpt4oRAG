
![](https://miro.medium.com/v2/resize:fit:1400/1*eMb_xmKGM1rcC4zPgZA1NQ.png)

## Project Overview

This project demonstrates the use of the LangChain library to build a Retrieval-Augmented Generation (RAG) pipeline. The pipeline involves loading data from a web source, creating vector embeddings for the data, and building a retriever for question-answering tasks using a language model. The goal is to enhance the capability of language models to provide accurate and contextually relevant answers by leveraging external knowledge sources.

## Prerequisites

Before running the project, ensure you have the following prerequisites installed:

- Python 3.10 or higher
- `dotenv` library
- `langchain` library
- `langchain_openai` library
- `langchain_core` library
- `openai` library

## Setup

1. **Clone the repository** (if applicable):

   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install the required Python packages**:

   ```sh
   pip install python-dotenv langchain langchain_openai langchain_core openai
   ```

3. **Set up environment variables**:

   Create a `.env` file in the project directory with the following content:

   ```env
   OPENAI_API_KEY=<your_openai_api_key>
   ```

## Usage

1. **Load environment variables**:

   ```python
   from dotenv import load_dotenv
   import os

   load_dotenv()
   os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
   ```

2. **Load data from a web source**:

   ```python
   from langchain_community.document_loaders import WebBaseLoader

   loader = WebBaseLoader("https://en.wikipedia.org/wiki/Roman_Empire")
   data = loader.load()
   ```

3. **Create vector embeddings**:

   ```python
   from langchain_openai import OpenAIEmbeddings
   from langchain_objectbox.vectorstores import ObjectBox

   vector = ObjectBox.from_documents(data, OpenAIEmbeddings(), embedding_dimensions=768)
   ```

4. **Build the RAG pipeline**:

   ```python
   from langchain_openai import ChatOpenAI
   from langchain_core.output_parsers import StrOutputParser
   from langchain_core.prompts import ChatPromptTemplate
   from langchain.chains import RetrievalQA
   from langchain import hub

   llm = ChatOpenAI(model="gpt-4o")
   prompt = hub.pull("rlm/rag-prompt")
   qa_chain = RetrievalQA.from_chain_type(
       llm,
       retriever=vector.as_retriever(),
       chain_type_kwargs={"prompt": prompt}
   )
   ```

5. **Ask questions and retrieve answers**:

   ```python
   question = "What was the government like in Roman Empire?"
   result = qa_chain({"query": question })
   print(result["result"])
   ```

   ```python
   question = "How was the education in Roman Empire?"
   result = qa_chain({"query": question })
   print(result["result"])
   ```

## Example Output

1. **Question**: What was the government like in Roman Empire?

   **Answer**:
   ```
   The Roman Empire's government consisted of three major elements: the central government, the military, and the provincial government. The central government included elected magistrates and appointed officials who managed different aspects of state affairs, while the military maintained control and order. Provincial governance involved local elites and officials to ensure efficient administration and taxation, often allowing local laws to coexist with Roman law.
   ```

2. **Question**: How was the education in Roman Empire?

   **Answer**:
   ```
   Education in the Roman Empire was moral and practical, aimed at instilling Roman values and often provided by parents or through apprenticeships. Formal education, available primarily to those who could afford it, included primary schooling in reading, writing, and arithmetic, with higher education focusing on literature, rhetoric, and philosophy. Schools became more numerous during the Empire, and education was considered essential for career advancement and social mobility, particularly for the urban elites.
   ```

## Additional Notes

- Ensure you have the correct API key for OpenAI and that it is properly loaded into the environment variables.
- The LangChain library and its components are subject to updates; refer to the official documentation for the latest features and changes.

---

This README provides a comprehensive guide to setting up and using the RAG pipeline for question-answering tasks using LangChain. Feel free to modify and expand the README as needed for your specific use case.
