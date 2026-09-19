"""
Streamlit Web 界面：本地文档 RAG 问答演示
运行：streamlit run web_rag.py
"""
import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="混合检索 RAG 知识库问答", page_icon="📖", layout="wide")
st.title("📖 基于 LangChain 的混合检索 RAG 私有知识库问答系统")
st.caption("向量检索 + BM25 关键词检索（RRF 融合）→ BGE-Reranker 精排 → LLM 生成")


@st.cache_resource(show_spinner="正在加载模型与向量库（首次需下载 Embedding / Reranker 模型）...")
def load_rag():
    return RAGPipeline(doc_path="doc.txt", persist_dir="./chroma_db")


rag = load_rag()

question = st.text_input("请输入你的问题：", placeholder="AI应用开发实习需要掌握哪些技术？")

if question:
    with st.spinner("正在混合检索、精排并生成回答..."):
        out = rag.answer(question)

    st.subheader("🤖 回答")
    st.write(out["answer"])

    st.subheader("📚 检索到的参考片段（已 Rerank 精排）")
    for i, c in enumerate(out["contexts"], 1):
        with st.expander(f"片段 {i}"):
            st.write(c)
