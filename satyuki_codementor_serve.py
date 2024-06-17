import os, os.path as osp
import sys
import argparse
from tqdm import tqdm
import glob
import gradio as gr
import time
import asyncio
import threading

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnablePick, RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


async def astream(invocable, inputs, *args, **kwargs):
    async for chunk in invocable.astream(inputs, *args, **kwargs):
        print(chunk, end="")

llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct",
    nvidia_api_key="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",
    temperature=0.6
)
embedder = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")

code_interpretation_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an AI agent tasked with understanding chunks from a codebase that is written in C++ and CUDA. Keep the following points in mind while writing interpretation:
• Note down and briefly describe any headers, macros, or #defines present and their purposes.
• Identify and list the class names, along with all member functions and variables in the chunk of code.
• For each function, note its purpose and role within the code.
• For each member variable of a class, note where it is referred to and its purpose.
• All responses MUST be in bullet points - must start with the character '•'
• Do NOT use any other sentence formatting.
• Do NOT include any introductory or concluding statements."""),
    ("user", "The code chunk you will need to process is:\n{code_chunk}")
])

code_interpretation_chain = code_interpretation_prompt_template | llm

combined_interpretation_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an AI agent tasked with combining the essence of the below two code inferences:
• Combine the essence of two code inferences into a single, standalone passage.
• Treat function or class interpretations that spill over as a single logical unit.
• Include all unique information from both summaries.
• Use concise technical language.
• Ensure consistency in terminology and formatting.
• All responses MUST be in bullet points - must start with the character '•'
• Do NOT use any other sentence formatting.
• Do NOT include any introductory or concluding statements.
"""),
    ("user", "The two code inferences you will need to cobine are:\n text 1:{summary_1} \n text 2:{summary_2}")
])

combined_interpretation_chain = combined_interpretation_prompt_template | llm


class KnowledgeBase:
    def __init__(self, repo_path):
        self.db = Chroma(embedding_function=embedder, persist_directory=osp.join(".", "databases", osp.basename(repo_path)))
        self.language_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size=1000, chunk_overlap=100)
        self.repo_path = repo_path
        self.in_memory_db = []
        self.per_file_interpretations = []

    def add_knowledge_per_file(self, file_path):
        print(f"Processing {file_path}...")
        interpretations_per_chunk = []
        loader = TextLoader(file_path)
        file_data = loader.load()
        code_chunks = self.language_splitter.split_documents(file_data)

        for chunk in tqdm(code_chunks):
            interpretation = code_interpretation_chain.invoke(chunk.page_content)
            print("****************************")
            print(interpretation)
            print("****************************")
            source_file_name = chunk.metadata["source"]
            interpretations_per_chunk.append(Document(page_content=source_file_name+"\n"+interpretation, metadata=chunk.metadata))
        
        read_buffer = interpretations_per_chunk.copy()
        merge_buffer = []

        while len(read_buffer) > 1:
            for i in range(0, len(read_buffer), 2):
                if i == len(read_buffer)-1:
                    merge_buffer.append(Document(page_content=read_buffer[i].page_content, metadata=read_buffer[i].metadata))
                else:
                    interpretation = combined_interpretation_chain.invoke({"summary_1":read_buffer[i].page_content, "summary_2":read_buffer[i+1].page_content})
                    source_file = read_buffer[i].metadata["source"]
                    print("****************************")
                    print(interpretation)
                    print("****************************")
                    merge_buffer.append(Document(page_content=source_file+"\n"+interpretation, metadata=read_buffer[i].metadata))

            read_buffer.clear()
            read_buffer.extend(merge_buffer)
            self.in_memory_db.extend(merge_buffer)
            merge_buffer.clear()

        self.per_file_interpretations.append(read_buffer[0])

    def generate_knowledge_base(self):
        source_files = glob.glob(osp.join(self.repo_path, "*.*"))

        for source_file in source_files:
            self.add_knowledge_per_file(source_file)
        
        read_buffer = self.per_file_interpretations.copy()
        merge_buffer = []

        while len(read_buffer) > 1:
            for i in range(0, len(read_buffer), 2):
                if i == len(read_buffer)-1:
                    merge_buffer.append(Document(page_content=read_buffer[i].page_content, metadata=read_buffer[i].metadata))
                else:
                    interpretation = combined_interpretation_chain.invoke({"summary_1":read_buffer[i].page_content, "summary_2":read_buffer[i+1].page_content})
                    source_list_set = set(read_buffer[i].metadata["source"].split("|")).union(read_buffer[i+1].metadata["source"].split("|"))
                    source_list_str = "|".join(source_list_set)
                    combined_meta = {"source": source_list_str}

                    merge_buffer.append(Document(page_content=source_list_str+"\n"+interpretation, metadata=combined_meta))

            read_buffer.clear()
            read_buffer.extend(merge_buffer)
            self.in_memory_db.extend(merge_buffer)
            merge_buffer.clear()

        import pickle

        with open(osp.join(".", "temp", "in_memory_db_new.pkl", "wb")) as pkl_file:
            pickle.dump(self.in_memory_db, pkl_file)
        self.db.add_documents(self.in_memory_db)

db_path = osp.join(".", "databases", "knowledge_db", "vector_store")
db = Chroma(embedding_function=embedder, persist_directory=db_path)


qa_chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that is tasked with having a conversation with a user who asks queries related to a code base that you have understood thoroughly. You are well-versed in C++ and CUDA. The queries might involve details such as what a specific function does, the purpose of a line of code, how a component fits into the architecture, or any other technical aspect. Answer each query with the help of the provided supporting context."),
    MessagesPlaceholder("history"),
    ("human", "Answer the query below using the provided context:\nQuery:{query}\nContext:{context}")
])

info_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 5}
)
preproc_chain = {
    "history": RunnableLambda(lambda inp: inp["history"]),
    "query": RunnableLambda(lambda inp: inp["query"]),
    "context": (RunnablePick(keys="query") | info_retriever)
}
qa_chain = (preproc_chain | qa_chat_prompt_template | llm)


# ! GRADIO

def build_knowledge_base(code_base_root):
    gr.Info("Processing code base...")
    kbase = KnowledgeBase(code_base_root)
    knowledge_creation_thread = threading.Thread(target=lambda: kbase.generate_knowledge_base(), name="Build Knowledge Base")
    knowledge_creation_thread.start()

async def chat_bot_query(message, history):
    if message is None or message == "":
        yield "Input cannot be blank."
    else:
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(('human', human))
            history_langchain_format.append(('ai', ai))

        inputs = {"history": history_langchain_format, "query": message}
        # yield str({key: runnable.invoke(inputs) for key, runnable in preproc_chain.items()})
        response = ""
        async for chunk in qa_chain.astream(inputs):
            response += chunk.content
            yield response

demo = gr.TabbedInterface(
    [
        gr.Interface(
            fn=build_knowledge_base,
            inputs=[gr.Textbox(label="Path to code base root folder")],
            outputs=[],
            allow_flagging='never',
            submit_btn="Build Knowledge Base"
        ),
        gr.ChatInterface(
            fn=chat_bot_query,
            chatbot=gr.Chatbot(label="Ask Satyuki", height="100%", show_copy_button=True)
        )
    ],
    ["Build Knowledge Base", "Ask Questions"],
    title="Satyuki, the Code Mentor",
    css="height: 100%"
)
demo.launch(server_port=8000)
