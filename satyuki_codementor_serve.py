import os, os.path as osp
import sys
import argparse
from tqdm import tqdm
import glob

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document


async def astream(invocable, inputs, *args, **kwargs):
    async for chunk in invocable.astream(inputs, *args, **kwargs):
        print(chunk, end="")

llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct",
    nvidia_api_key="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",
    temperature=0.6
)
gpt4all_embd = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")

info_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI agent tasked with processing chunks from a codebase. Your objective is to thoroughly understand the information contained in each chunk and return a CONCISE summary. You must provide your answer strictly in BULLET POINTS - all output lines must start with the character '•'. Do NOT use any other sentence formatting, and do NOT include any introductory or concluding statements."),
    ("user", "The code chunk you will need to process is:\n{code_chunk}")
])
info_chain = info_prompt_template | llm

hierarchy_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an AI agent tasked with combining 2 code summaries. Your task is to merge two summaries of different code base parts into a single, clear, and comprehensive knowledge base. Combine overlapping details, organize content logically by grouping related functionalities, include all unique information from both summaries, use concise technical language, maintain context and purpose of each code chunk, and ensure consistency in terminology and formatting.You must provide your response strictly in BULLET POINTS - all output lines must start with the character '•'. Do NOT use any other sentence formatting. Do NOT include any introductory or concluding statements. DO NOT use the words like - 'summary' or 'chunk' in your response.
"""),
    ("user", "The two summaries you will need to process are:\n summary 1:{summary_1} \n summary 2:{summary_2}")
])

hierarchy_chain = hierarchy_prompt_template | llm

class KnowledgeBase:
    def __init__(self, repo_path):
        self.db = Chroma(embedding_function=embedder, persist_directory=".\\db\\")
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
            interpretation = info_chain.invoke(chunk.page_content)
            interpretation_pointwise = parse_bullet_points(interpretation)
            print("****************************")
            print(interpretation_pointwise)
            print("****************************")
            source_file_name = chunk.metadata["source"]
            interpretations_per_chunk.append(Document(page_content=source_file_name+"\n"+interpretation_pointwise, metadata=chunk.metadata))
        
        read_buffer = interpretations_per_chunk.copy()
        merge_buffer = []

        while len(read_buffer) > 1:
            for i in range(0, len(read_buffer), 2):
                if i == len(read_buffer)-1:
                    merge_buffer.append(Document(page_content=read_buffer[i].page_content, metadata=read_buffer[i].metadata))
                else:
                    interpretation = hierarchy_chain.invoke({"summary_1":read_buffer[i].page_content, "summary_2":read_buffer[i+1].page_content})
                    interpretation_pointwise = parse_bullet_points(interpretation)
                    source_file = read_buffer[i].metadata["source"]
                    print("****************************")
                    print(interpretation_pointwise)
                    print("****************************")
                    merge_buffer.append(Document(page_content=source_file+"\n"+interpretation_pointwise, metadata=read_buffer[i].metadata))

            read_buffer.clear()
            read_buffer.extend(merge_buffer)
            self.in_memory_db.extend(merge_buffer)
            merge_buffer.clear()

        self.per_file_interpretations.append(read_buffer[0])

    def generate_knowledge_base(self):
        source_files = glob.glob(self.repo_path + "\\*.*")
        for source_file in source_files:
            self.add_knowledge_per_file(source_file)
        
        read_buffer = self.per_file_interpretations.copy()
        merge_buffer = []

        while len(read_buffer) > 1:
            for i in range(0, len(read_buffer), 2):
                if i == len(read_buffer)-1:
                    merge_buffer.append(Document(page_content=read_buffer[i].page_content, metadata=read_buffer[i].metadata))
                else:
                    interpretation = hierarchy_chain.invoke({"summary_1":read_buffer[i].page_content, "summary_2":read_buffer[i+1].page_content})
                    interpretation_pointwise = parse_bullet_points(interpretation)
                    source_list_set = set(read_buffer[i].metadata["source"].split("|")).union(read_buffer[i+1].metadata["source"].split("|"))
                    source_list_str = "|".join(source_list_set)
                    combined_meta = {"source": source_list_str}

                    merge_buffer.append(Document(page_content=source_list_str+"\n"+interpretation_pointwise, metadata=combined_meta))

            read_buffer.clear()
            read_buffer.extend(merge_buffer)
            self.in_memory_db.extend(merge_buffer)
            merge_buffer.clear()

        self.db.add_documents(self.in_memory_db)

repo_path = ".\\PathTracerAP"

knowledge_base = KnowledgeBase(repo_path)
knowledge_base.generate_knowledge_base()


# ! GRADIO


import gradio as gr
import time

def process_user_query(message, history):
    if len(history) % 2 == 0:
        return f"Yes, I do think that '{message}'"
    else:
        return "I don't think so"

def process_source_code(dir_path):
    progress = gr.Progress()
    # Simulate file processing with a delay
    for i in range(10):
        time.sleep(0.5)
        progress(i / 10)
        a = True

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Satyuki, the Code Mentor")
    dir_path = gr.Textbox(label="Source code root directory", placeholder="Enter the path to the root folder of your source code.")
    process_button = gr.Button("Process Code Base")
    output_text = gr.Textbox(label="")
    
    process_button.click(fn=process_source_code, inputs=dir_path, outputs=output_text)

demo.launch()
gr.ChatInterface(process_user_query).launch()