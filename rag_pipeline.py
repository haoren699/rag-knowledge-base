import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
    def __init__(self, doc_path="doc.txt", persist_dir="./chroma_db"):
        self.llm = ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.1,
        )
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

        docs = TextLoader(doc_path, encoding="utf-8").load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        self.splits = splitter.split_documents(docs)
        self.chunk_texts = [d.page_content for d in self.splits]

        self.vectorstore = Chroma.from_documents(
            documents=self.splits,
            embedding=self.embedding,
            persist_directory=persist_dir,
        )
        self.bm25 = BM25Okapi([t.split() for t in self.chunk_texts])

    def vector_search(self, query, k=20):
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        idxs = []
        for doc, _ in results:
            try:
                idxs.append(self.chunk_texts.index(doc.page_content))
            except ValueError:
                pass
        return idxs

    def bm25_search(self, query, k=20):
        scores = self.bm25.get_scores(query.split())
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    @staticmethod
    def rrf_fuse(rank_lists, k=60):
        scores = {}
        for ranks in rank_lists:
            for rank, idx in enumerate(ranks):
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def hybrid_retrieve(self, query, top_k=5):
        v_rank = self.vector_search(query)
        b_rank = self.bm25_search(query)
        fused = self.rrf_fuse([v_rank, b_rank])[:20]

        pairs = [(query, self.chunk_texts[i]) for i in fused]
        rerank_scores = self.reranker.predict(pairs)
        order = sorted(range(len(fused)), key=lambda i: rerank_scores[i], reverse=True)
        return [self.chunk_texts[fused[i]] for i in order[:top_k]]

    def answer(self, question):
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
