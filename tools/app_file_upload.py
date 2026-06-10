import streamlit as st

from knowledge_base import KnowledgeBaseService

st.title("知识库更新")

file = st.file_uploader("上传文件", type=["txt", "docx"],accept_multiple_files=False)

if "knowledge_base" not in st.session_state:
    st.session_state["knowledge_base"] = KnowledgeBaseService()

if file is not None:
    file_name = file.name
    file_type = file.type
    file_content = file.getvalue().decode("utf-8")
    st.write(f"文件名: {file_name}")
    st.write(f"文件类型: {file_type}")
    st.write(f"文件内容: {file_content}")


    state = st.session_state["knowledge_base"].upload_by_str(file_content, file_name)
    st.write(state)

st.write(st.session_state["knowledge_base"].chroma.similarity_search("李孟"))