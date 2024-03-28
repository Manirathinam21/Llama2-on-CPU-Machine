from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from src.helper import *

#Load the PDF File
loader= DirectoryLoader('data/', glob='*.pdf', loader_cls= PyPDFLoader)
documents= loader.load()

#Split the Text into Chunks
text_splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
text_chunks= text_splitter.split_documents(documents)

#Load the Embedding Model
embeddings= HuggingFaceEmbeddings( model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                  model_kwargs={'device':'cpu'})
                                  
#Convert the Text Chunks into Embeddings and Create a FAISS Vector Store   
vector_db= FAISS.from_documents(text_chunks, embeddings) 


#Loading the Opensource LLM Llama-2-7B-chat-hf model
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type='llama',
                    config={'max_new_tokens':128,
                          'temperature':0.01}
                    )

#promptTemplate
qa_prompt = PromptTemplate(input_variables=['context', 'question'], template= template)


# Retrivel Chain to generate the output using Llama-2 llm for given question
chain= RetrievalQA.from_chain_type(llm=llm,
                        chain_type='stuff',
                        retriever= vector_db.as_retriever(search_kwargs={'k':2}),
                        return_source_documents=False,
                        chain_type_kwargs={'prompt': qa_prompt}
                        )

user_input = "Tell me about Rainfall Measurement of the paper"

response= chain({'query':user_input})
print(f"Answer:{response['result']}")
                                                              