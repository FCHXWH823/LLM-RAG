from langchain_community.document_loaders import PyMuPDFLoader
import pandas as pd
import csv

# Load your PDFs
pdf_loader = PyMuPDFLoader("/Users/fch/Python/LLM-RAG/VerilogTextBooks/CernyDudani-SVA- The Power of Assertions in SystemVerilog.pdf")
documents = pdf_loader.load()

# If you have multiple PDFs, you can load them in a loop
# pdf_paths = ["path/to/doc1.pdf", "path/to/doc2.pdf"]
# documents = []
# for path in pdf_paths:
#     loader = PyMuPDFLoader(path)
#     documents.extend(loader.load())

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split the PDF text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key="")

# Index the document chunks in a vector store
vector_store = FAISS.from_texts([chunk.page_content for chunk in chunks], embeddings)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    api_key=""
    )

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

df = pd.read_csv('asserted-verilog-evaluation-dataset.csv')
# Run a query
with open('Openai-4o-mini-Prompted-Assertion-Generation-Results.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['code','HumanExplanation','pure code','prompt','llm_response'])
    for id in range(len(df)):
        code = df.iloc[id]['code']
        humanexplanation = df.iloc[id]['HumanExplanation']
        purecode = df.iloc[id]['pure code']
        prompt = f"Given Verilog code snippet as below: \n{purecode}\n Please generate a rewritten version of it, which contains several useful assertions and each of them follows a description as follows:{humanexplanation}\n"
        

        llm_response = qa_chain.run(prompt)

        csv_writer.writerow([code,humanexplanation,purecode,prompt,llm_response])
