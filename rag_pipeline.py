"""
RAG 完整链路：文档加载 -> 切分 -> 向量化 -> 向量+BM25混合检索 -> BGE-Reranker精排 -> LLM生成
对应简历：基于 LangChain 的混合检索 RAG 私有知识库问答系统
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内 HuggingFace 镜像

from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    def __init__(self, doc_path: str = "doc.txt", persist_dir: str = "./chroma_db"):
        # ---------- LLM（云端 DeepSeek） ----------
        self.llm = ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.1,
        )

        # ---------- 本地 Embedding 模型 ----------
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ---------- 本地 Reranker（cross-encoder） ----------
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

        # ---------- 加载并切分文档 ----------
        loader = TextLoader(doc_path, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        self.splits = splitter.split_documents(docs)
        self.chunk_texts = [d.page_content for d in self.splits]

        # ---------- 向量库 ----------
        self.vectorstore = Chroma.from_documents(
            documents=self.splits,
            embedding=self.embedding,
            persist_directory=persist_dir,
        )

        # ---------- BM25 关键词检索器 ----------
        tokenized = [t.split() for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

    # ========== 三路检索 ==========
    def vector_search(self, query: str, k: int = 20) -> List[int]:
        """向量检索，返回 chunk 下标列表"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        idxs = []
        for doc, _ in results:
            try:
                idxs.append(self.chunk_texts.index(doc.page_content))
            except ValueError:
                continue
        return idxs

    def bm25_search(self, query: str, k: int = 20) -> List[int]:
        """BM25 关键词检索，返回 chunk 下标列表"""
        scores = self.bm25.get_scores(query.split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return top

    @staticmethod
    def rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> List[int]:
        """Reciprocal Rank Fusion：把多路检索结果按排名融合"""
        scores = {}
        for ranks in rank_lists:
            for rank, idx in enumerate(ranks):
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """混合检索 -> Reranker 精排 -> 返回 top_k 段"""
        v_rank = self.vector_search(query, k=20)
        b_rank = self.bm25_search(query, k=20)
        fused = self.rrf_fuse([v_rank, b_rank])[:20]

        # Reranker 精排
        pairs = [(query, self.chunk_texts[i]) for i in fused]
        rerank_scores = self.reranker.predict(pairs)
        order = sorted(range(len(fused)), key=lambda i: rerank_scores[i], reverse=True)
        return [self.chunk_texts[fused[i]] for i in order[:top_k]]

    # ========== 生成回答 ==========
    def answer(self, question: str) -> dict:
        contexts = self.hybrid_retrieve(question, top_k=3)
        context_text = "\n---\n".join(contexts)
        prompt = f"""你是文档问答助手，严格基于下面参考资料回答用户问题。
如果参考资料中没有相关信息，直接回答"文档中没有相关信息"，禁止编造。

参考资料：
{context_text}

用户问题：{question}
"""
        resp = self.llm.invoke(prompt)
        return {"answer": resp.content, "contexts": contexts}


if __name__ == "__main__":
    rag = RAGPipeline()
    out = rag.answer("AI应用开发实习需要掌握哪些技术？")
    print("【回答】", out["answer"])
    print("\n【检索片段】")
    for i, c in enumerate(out["contexts"], 1):
        print(f"[{i}] {c[:80]}...")
