from tkinter import X
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.llms import Tongyi
from langchain_core.callbacks import ChainManagerMixin
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.runnables import RunnableLambda

myfunc = RunnableLambda(lambda x: {"english_word": x.content})


model = ChatTongyi(
    model="qwen-plus",
    api_key="sk-37c32105b3674024a90a612856c6a4ad",
    streaming=True
)

first_prompt = PromptTemplate.from_template("请帮我把中文翻译为英文：{chinese_word}")

second_prompt = PromptTemplate.from_template("请帮我把英文翻译为日文，如果输入不是英文则返回“不是英文”：{english_word}")


chain = first_prompt | model | myfunc | second_prompt | model
first_response = chain.stream({"chinese_word": "你好"})
print(first_response)
print(type(first_response))
for chunk in first_response:
    print(chunk.content, end="", flush=True)
print("\n")

# model = Tongyi(
#     model="qwen-plus",
#     api_key="sk-37c32105b3674024a90a612856c6a4ad",
#     streaming=True
# )

# first_prompt = PromptTemplate.from_template("请帮我把{chinese_word}翻译为英文,把结果填成下面的格式：\"帮我把output翻译为日文\",其中output为翻译后的英文单词，举例：输入：你好，输出：帮我把hello翻译为日文")



# chain = first_prompt | model |model

# response = chain.stream({"chinese_word": "你好"})
# for chunk in response:
#     print(chunk, end="", flush=True)
# print("\n")
