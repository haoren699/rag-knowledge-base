"""
Recall@k 评估脚本：对比 纯向量 / 混合检索 / 混合+Rerank 三种配置的召回率
对应简历：基于 Recall@k 指标开展消融实验
用法：python eval_recall.py
"""
import json
from rag_pipeline import RAGPipeline


def load_test_set(path: str = "test_qa.jsonl"):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def recall_at_k(retrieved_chunks, answer_keyword: str, k: int) -> int:
    """前 k 段里有没有任一段包含 answer_keyword"""
    for chunk in retrieved_chunks[:k]:
        if answer_keyword in chunk:
            return 1
    return 0


def main():
    rag = RAGPipeline(doc_path="doc.txt", persist_dir="./chroma_db")
    test_set = load_test_set()
    print(f"加载测试集 {len(test_set)} 条\n")

    for k in [3, 5]:
        only_vec_hit = 0
        hybrid_hit = 0
        hybrid_rerank_hit = 0

        for item in test_set:
            q = item["question"]
            kw = item["answer_keyword"]

            # 配置1：纯向量 top-k
            vec_idx = rag.vector_search(q, k=k)
            vec_chunks = [rag.chunk_texts[i] for i in vec_idx]
            only_vec_hit += recall_at_k(vec_chunks, kw, k)

            # 配置2：混合检索（无 Rerank）top-k
            v_rank = rag.vector_search(q, k=20)
            b_rank = rag.bm25_search(q, k=20)
            fused = rag.rrf_fuse([v_rank, b_rank])[:k]
            hybrid_chunks = [rag.chunk_texts[i] for i in fused]
            hybrid_hit += recall_at_k(hybrid_chunks, kw, k)

            # 配置3：混合检索 + Rerank top-k
            contexts = rag.hybrid_retrieve(q, top_k=k)
            hybrid_rerank_hit += recall_at_k(contexts, kw, k)

        n = len(test_set)
        print(f"===== Recall@{k} =====")
        print(f"  纯向量检索            : {only_vec_hit}/{n} = {only_vec_hit/n:.1%}")
        print(f"  混合检索（向量+BM25）  : {hybrid_hit}/{n} = {hybrid_hit/n:.1%}")
        print(f"  混合检索 + BGE-Rerank  : {hybrid_rerank_hit}/{n} = {hybrid_rerank_hit/n:.1%}")
        print()


if __name__ == "__main__":
    main()
