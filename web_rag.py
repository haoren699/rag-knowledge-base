import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG知识库问答", page_icon="📖", layout="wide")
st.title("📖 本地文档 RAG 问答系统")
st.caption("向量 + BM25 混合检索 → BGE-Reranker 精排 → LLM 生成")


@st.cache_resource(show_spinner="加载模型中...")
def load_rag():
    return RAGPipeline()


rag = load_rag()
question = st.text_input("输入问题：", placeholder="AI应用开发实习需要掌握哪些技术？")

if question:
    with st.spinner("检索并生成中..."):
        out = rag.answer(question)

    st.subheader("回答")
    st.write(out["answer"])

    st.subheader("检索到的片段")
    for i, c in enumerate(out["contexts"], 1):
        with st.expander(f"片段 {i}"):
            st.write(c)
