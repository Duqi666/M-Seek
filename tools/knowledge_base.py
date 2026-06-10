import os
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from datetime import datetime
def check_md5(md5_str: str,md5_file: str):
    #检查传入的md5字符串是否已经被处理过了
    if md5_str is not None:
        if not os.path.exists(md5_file):
            print(f"md5文件 {md5_file} 不存在")
            return False
        else:
            for line in open(md5_file, "r").readlines():
                if line.strip() == md5_str:
                    return True
            return False
    else:
        print("md5文件 is None")


def save_md5(file_path, md5):
#将md5字符串保存到文件中
    with open(file_path, "a") as f:
        f.write(md5 + "\n")

import hashlib
def get_md5(input_str):
#从文件中读取md5字符串
    md5 = hashlib.md5(input_str.encode("utf-8")).hexdigest()
    return md5


class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.PERSIST_DIRECTORY, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4",dashscope_api_key="sk-37c32105b3674024a90a612856c6a4ad"),
            persist_directory=config.PERSIST_DIRECTORY
            )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
            length_function=len
        )
        
 
    def upload_by_str(self, data: str, file_name: str):
        data_md5 = get_md5(data)
        if check_md5(data_md5, config.MD5_FILE):
            
            return f"数据 {data_md5} 已被处理过了"
        else:
            print(f"数据 {data_md5} 未被处理过")
            if len(data) > config.SPLIT_NUM_SIZE:
                knowledge_chunks: list[str] = self.text_splitter.split_text(data)
            else:
                knowledge_chunks: list[str] = [data]
        
        metadata = {"source": "knowledge_base", "create_time": datetime.now(),"file_name": file_name} 

        self.chroma.add_texts(knowledge_chunks, metadata=[metadata]*len(knowledge_chunks))
        save_md5(config.MD5_FILE, data_md5)
        return f"数据 {data_md5} 处理完成"

