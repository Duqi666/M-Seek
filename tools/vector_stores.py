from langchain_core.load import serializable
from langchain_chroma import Chroma
import config
from langchain_community.embeddings import DashScopeEmbeddings



class VectorStoreService:
    def __init__(self,embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embedding,
            persist_directory=config.PERSIST_DIRECTORY
            )
    def get_retriever(self):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": config.K})

def __main__():
    vector_store_service = VectorStoreService(embedding=DashScopeEmbeddings(model="text-embedding-v4",dashscope_api_key=config.API_KEY))
    retriever = vector_store_service.get_retriever()
    print(retriever.invoke("你好"))
    print(11111)