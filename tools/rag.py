import warnings

# 屏蔽 langchain-community 弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
# 屏蔽 requests 字符编码依赖警告
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings

class RAGService:
    def __init__(self):
        self.vector_store_service = VectorStoreService(
            embedding=DashScopeEmbeddings(
                model="text-embedding-v4",
                dashscope_api_key=config.API_KEY
                )
            )
        self.prompt = PromptTemplate.from_template("根据以下上下文回答问题：{context}\n问题：{question}")
        self.chat_model = ChatTongyi(
            model=config.MODEL_NAME,
            api_key=config.API_KEY,
            streaming=True
        )

        self.retriever= self.vector_store_service.get_retriever()


def __main__():
    rag_service = RAGService()
    print(rag_service.retriever.invoke("你好"))