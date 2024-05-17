
![](https://miro.medium.com/v2/resize:fit:1400/1*eMb_xmKGM1rcC4zPgZA1NQ.png)


# Project Overview

This project demonstrates the use of the LangChain library to build multiple Retrieval-Augmented Generation (RAG) pipelines. The pipelines involve loading data from web sources and local repositories, creating vector embeddings for the data, and building retrievers for question-answering tasks using a language model. The goal is to enhance the capability of language models to provide accurate and contextually relevant answers by leveraging external knowledge sources.

## Prerequisites

Before running the project, ensure you have the following prerequisites installed:

- Python 3.10 or higher
- `dotenv` library
- `langchain` library
- `langchain_openai` library
- `langchain_core` library
- `openai` library
- `fitz` (PyMuPDF) library

## Setup

1. **Clone the repository** (if applicable):

   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install the required Python packages**:

   ```sh
   pip install python-dotenv langchain langchain_openai langchain_core openai PyMuPDF
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

   loader1 = WebBaseLoader("https://en.wikipedia.org/wiki/Roman_Empire")
   data1 = loader1.load()
   ```

3. **Extract text from a local PDF file**:

   Chunking is an important preprocessing step when working with text data, especially in tasks involving embeddings and retrieval-augmented generation (RAG) pipelines. Here are the reasons and the process behind chunking:

### Why Chunking is Necessary:

1. **Model Input Limitations**:
   - **Token Limits**: Language models, including those used for embeddings, have a maximum limit on the number of tokens (words or subwords) they can process in a single input. If a document exceeds this limit, it cannot be processed in one go.
   - **Memory and Computation**: Handling very large documents in one piece can be computationally expensive and memory-intensive. Breaking them into smaller chunks makes the process more efficient and manageable.

2. **Improving Retrieval Accuracy**:
   - **Granularity**: Smaller chunks allow for more granular retrieval, meaning that the retriever can find and return the most relevant parts of the document more accurately. If documents were not chunked, a query might retrieve a very large and less specific part of the document, reducing the precision of the answer.
   - **Context**: Each chunk can be embedded separately, preserving more context and nuance than trying to embed very large text blocks where detailed information might be lost.

3. **Enhanced Performance**:
   - **Parallel Processing**: Smaller chunks can be processed in parallel, speeding up both the embedding and retrieval processes.
   - **Focused Attention**: When generating embeddings or processing queries, models can focus on smaller, more coherent pieces of text, improving the relevance and quality of the embeddings and subsequent answers.

### How Chunking is Done:

1. **Loading the Text**:
   - Text is extracted from sources such as web pages or PDF documents.

2. **Splitting into Chunks**:
   - The extracted text is split into smaller, manageable chunks based on a predefined maximum chunk size (e.g., 1000 tokens). This can be done at sentence boundaries, paragraphs, or other logical breaks to ensure chunks are coherent and self-contained.

### Example Code for Chunking:

```python
import fitz  # PyMuPDF
from langchain_core.schema import Document

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def split_text(text, max_chunk_size=1000):
    words = text.split()
    chunks = []
    chunk = []
    chunk_size = 0
    for word in words:
        if chunk_size + len(word) + 1 <= max_chunk_size:
            chunk.append(word)
            chunk_size += len(word) + 1
        else:
            chunks.append(" ".join(chunk))
            chunk = [word]
            chunk_size = len(word) + 1
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

pdf_path = '/path/to/human-nutrition-text.pdf'
document_content = extract_text_from_pdf(pdf_path)
document_chunks = split_text(document_content)

documents = [Document(page_content=chunk) for chunk in document_chunks]
```

### Workflow with Chunking:

1. **Extract Text**: Load the full text from the document.
2. **Split Text**: Split the full text into smaller chunks that are within the token limit of the model.
3. **Create Documents**: Convert each chunk into a document object.
4. **Embed and Store**: Generate embeddings for each chunk and store them in a vector store.
5. **Query Processing**: When a query is made, retrieve relevant chunks based on their embeddings, ensuring accurate and contextually relevant responses.

By chunking the text, you ensure that the processing stays within the limits of the models being used, improves the efficiency and accuracy of retrieval, and ultimately enhances the performance of the RAG pipeline.

5. **Create vector embeddings**:

   ```python
   from langchain_openai import OpenAIEmbeddings
   from langchain_objectbox.vectorstores import ObjectBox

   vector1 = ObjectBox.from_documents(data1, OpenAIEmbeddings(), embedding_dimensions=768)
   vector2 = ObjectBox.from_documents(documents, OpenAIEmbeddings(), embedding_dimensions=768)
   ```
   Embeddings are a crucial part of many modern natural language processing (NLP) and information retrieval tasks because they enable the conversion of text into numerical representations that capture the semantic meaning of the text. Here’s why embeddings were used in this project:

### Why Use Embeddings in RAG Pipelines:

1. **Semantic Understanding**:
   - **Text Representation**: Embeddings transform words, sentences, or entire documents into fixed-size numerical vectors. These vectors capture the semantic meaning of the text, allowing similar pieces of text to have similar vector representations.
   - **Context Capture**: Modern embedding techniques, such as those provided by OpenAI, capture the context in which words are used, leading to better understanding and more accurate responses.

2. **Efficient Retrieval**:
   - **Similarity Search**: By embedding both the query and the documents into the same vector space, we can efficiently perform similarity searches to find the most relevant documents or text snippets for a given query.
   - **Vector Stores**: Using vector stores like ObjectBox allows for fast and efficient retrieval of the most relevant documents based on their embeddings.

3. **Enhanced Question Answering**:
   - **Context Augmentation**: In Retrieval-Augmented Generation (RAG) pipelines, embeddings are used to retrieve relevant documents that augment the context for a language model, improving the quality and accuracy of generated answers.
   - **Knowledge Integration**: By embedding external knowledge sources (like Wikipedia articles or PDF documents), we can integrate this knowledge into the language model’s responses, leading to more informative and accurate answers.

### How Embeddings Were Used in the Project:

1. **Loading Data**:
   - Data from web sources (e.g., Wikipedia articles) and local repositories (e.g., PDF documents) were loaded.

2. **Creating Vector Embeddings**:
   - Text data was converted into embeddings using OpenAI's embedding models. This involves passing the text through a model to obtain numerical vectors that represent the semantic meaning of the text.

3. **Building Vector Stores**:
   - The embeddings were stored in vector stores (ObjectBox in this case), allowing for efficient similarity search and retrieval.

4. **Retrieval-Augmented Generation (RAG) Pipelines**:
   - **Retriever**: When a query is made, the retriever component uses the embeddings to find the most relevant documents.
   - **QA Chain**: The retrieved documents augment the input to the language model, which then generates answers that are informed by the external knowledge embedded in those documents.

### Example Workflow:

1. **Query**: User asks, “What are complex carbs?”
2. **Embedding**: The query is embedded into a vector.
3. **Retrieval**: The vector is used to retrieve the most relevant document chunks from the vector store.
4. **Answer Generation**: The retrieved documents provide context for the language model, which generates a detailed and accurate answer.

By using embeddings, the RAG pipeline leverages both the language model's capabilities and external knowledge sources, leading to more accurate and contextually relevant answers.

5. **Build the first RAG pipeline**:

   ```python
   from langchain_openai import ChatOpenAI
   from langchain_core.output_parsers import StrOutputParser
   from langchain_core.prompts import ChatPromptTemplate
   from langchain.chains import RetrievalQA
   from langchain import hub

   llm1 = ChatOpenAI(model="gpt-4o")
   prompt1 = hub.pull("rlm/rag-prompt")
   qa_chain1 = RetrievalQA.from_chain_type(
       llm1,
       retriever=vector1.as_retriever(),
       chain_type_kwargs={"prompt": prompt1}
   )
   ```

6. **Build the second RAG pipeline**:

   ```python
   llm2 = ChatOpenAI(model="gpt-4o")
   prompt2 = hub.pull("rlm/rag-prompt")
   qa_chain2 = RetrievalQA.from_chain_type(
       llm2,
       retriever=vector2.as_retriever(),
       chain_type_kwargs={"prompt": prompt2}
   )
   ```

7. **Ask questions and retrieve answers from the first pipeline**:

   ```python
   question1 = "What was the government like in Roman Empire?"
   result1 = qa_chain1({"query": question1 })
   print(result1["result"])
   ```

   ```python
   question2 = "How was the education in Roman Empire?"
   result2 = qa_chain1({"query": question2 })
   print(result2["result"])
   ```

8. **Ask questions and retrieve answers from the second pipeline**:

   ```python
   question3 = "What are complex carbs?"
   result3 = qa_chain2({"query": question3 })
   print(result3["result"])
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

3. **Question**: What are complex carbs?

   **Answer**:
   ```
   Complex carbohydrates are long chains of monosaccharides that may be branched or unbranched. They include polysaccharides such as starches and fibers. Examples of foods rich in complex carbs are grains, legumes, and root vegetables like potatoes.
   ```

## Additional Notes

- Ensure you have the correct API key for OpenAI and that it is properly loaded into the environment variables.
- The LangChain library and its components are subject to updates; refer to the official documentation for the latest features and changes.

---

This README provides a comprehensive guide to setting up and using multiple RAG pipelines for question-answering tasks using LangChain. 
