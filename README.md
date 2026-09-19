# RAG 私有知识库问答系统

基于 LangChain 搭的一个本地 RAG 项目，练手用。

## 做了什么

- 文档切分用 RecursiveCharacterTextSplitter，chunk_size=512，overlap=64
- Embedding 用本地 all-MiniLM-L6-v2，向量库存 Chroma
- 检索做了向量 + BM25 混合，RRF 融合两路结果
- 接了 BGE-Reranker 做二次精排
- 用 Streamlit 搭了个网页界面
- 写了个简单的 Recall@k 脚本对比了纯向量 / 混合 / 混合+Rerank 的效果
- 另外把 RAG 检索封装成工具，接了个 ReAct Agent demo

## 技术栈

Python、LangChain、Chroma、sentence-transformers、rank-bm25、BGE-Reranker、Streamlit、DeepSeek API

## 跑起来

```bash
pip install -r requirements.txt
cp .env.example .env  # 填上自己的 DeepSeek key
streamlit run web_rag.py
```

评估：

```bash
python eval_recall.py
```

## 目录

```
rag_pipeline.py   # 核心链路
web_rag.py        # Streamlit 界面
agent_demo.py     # ReAct + RAG as Tool
eval_recall.py    # Recall@k 评估
doc.txt           # 测试文档
test_qa.jsonl     # 评估集
```
