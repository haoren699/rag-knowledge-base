# 基于 LangChain 的混合检索 RAG 私有知识库问答系统

一个完整可运行的 RAG（Retrieval-Augmented Generation）项目，实现了从文档加载、语义切分、向量检索、关键词检索、重排到 LLM 生成的完整链路，并附带 Recall@k 消融实验脚本与 Streamlit Web 演示界面。

## ✨ 项目特性

- **完整 RAG 链路**：文档加载 → 递归字符切分 → Embedding 向量化 → 向量+BM25 混合检索 → BGE-Reranker 精排 → 约束性 Prompt → LLM 生成
- **混合检索**：向量检索（语义召回）+ BM25（关键词精确匹配），使用 RRF（Reciprocal Rank Fusion）融合两路结果
- **BGE-Reranker 二次精排**：使用 `BAAI/bge-reranker-base` cross-encoder 对粗排结果重新打分，降低噪声上下文
- **消融实验**：自带 `eval_recall.py`，可对比纯向量 / 混合检索 / 混合+Rerank 三种配置的 Recall@k
- **ReAct Agent**：将 RAG 检索封装为工具，由 ReAct Agent 自主决定何时调用知识库
- **Web 演示**：Streamlit 一键启动，可交互式提问并查看检索到的参考片段

## 🛠 技术栈

| 模块 | 选型 |
|---|---|
| LLM | DeepSeek（OpenAI 兼容接口） |
| Embedding | sentence-transformers/all-MiniLM-L6-v2（本地） |
| 向量数据库 | Chroma |
| 关键词检索 | rank-bm25（BM25Okapi） |
| 重排模型 | BAAI/bge-reranker-base（cross-encoder） |
| 框架 | LangChain |
| Web | Streamlit |

## 📁 项目结构

```
.
├── rag_pipeline.py      # RAG 核心链路：混合检索 + Reranker + 生成
├── web_rag.py            # Streamlit Web 演示界面
├── agent_demo.py         # ReAct Agent（RAG 作为工具被调用）
├── eval_recall.py        # Recall@k 消融实验脚本
├── doc.txt               # 示例知识库文档
├── test_qa.jsonl         # 评估用问答集
├── requirements.txt      # Python 依赖
├── .env.example         # 环境变量模板
└── .gitignore
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 3. 命令行运行

```bash
python rag_pipeline.py
```

### 4. Web 演示

```bash
streamlit run web_rag.py
```

浏览器打开 `http://localhost:8501` 即可交互问答。

### 5. 运行消融实验

```bash
python eval_recall.py
```

会输出类似：

```
===== Recall@3 =====
  纯向量检索            : 8/12 = 66.7%
  混合检索（向量+BM25）  : 10/12 = 83.3%
  混合检索 + BGE-Rerank : 11/12 = 91.7%
```

## 🧠 核心设计

### 为什么用混合检索？

纯向量检索擅长语义相似，但对专有名词、精确编号匹配较弱；BM25 基于词频统计，对精确词命中非常敏感。两路结果通过 RRF 融合，取长补短。

### 为什么用 Reranker？

向量检索是双塔结构（问题和文档分别编码），快但精度有限；Reranker 是 cross-encoder，把问题和文档拼接后一起过模型，能捕捉细粒度相关性，准但慢。因此采用"向量粗排 top20 → Reranker 精排 top3"的工业界标准做法。

### 切分参数

- `chunk_size=512`，`chunk_overlap=64`
- 使用 `RecursiveCharacterTextSplitter`，按 `\n\n → \n → 。 → ！ → ？ → ； → 空格` 的优先级切分

## 📝 License

仅用于学习与个人作品集展示。
