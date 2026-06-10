from knowledge_base import KnowledgeBaseService
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


model = ChatTongyi(
    model="qwen-plus",
    api_key="sk-37c32105b3674024a90a612856c6a4ad",
    streaming=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "以我提供的参考知识为基础，回答我的问题，参考资料：{knowledge_chunks}"),
    ("human", "用户提问：{question}")
])
def print_prompt(prompt):
    print(prompt)
    return prompt
chain = prompt | print_prompt | model

knowledge_base = KnowledgeBaseService()
chroma = knowledge_base.chroma

question = "李孟轩是哪个国家的？"
knowledge_chunks = [i.page_content for i in chroma.similarity_search(question,2)]

print(knowledge_chunks)
response = chain.invoke({"knowledge_chunks": knowledge_chunks, "question": question})
print(response.content)
