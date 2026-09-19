import json
from rag_pipeline import RAGPipeline


def load_test_set(path="test_qa.jsonl"):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def hit(chunks, keyword, k):
    return int(any(keyword in c for c in chunks[:k]))


def main():
    rag = RAGPipeline()
    test_set = load_test_set()
    n = len(test_set)
    print(f"测试集 {n} 条\n")

    for k in [3, 5]:
        only_vec = hybrid = hybrid_rerank = 0
        for item in test_set:
            q, kw = item["question"], item["answer_keyword"]

            v = rag.vector_search(q, k=k)
            only_vec += hit([rag.chunk_texts[i] for i in v], kw, k)

            v_rank = rag.vector_search(q)
            b_rank = rag.bm25_search(q)
            fused = rag.rrf_fuse([v_rank, b_rank])[:k]
            hybrid += hit([rag.chunk_texts[i] for i in fused], kw, k)

            hybrid_rerank += hit(rag.hybrid_retrieve(q, top_k=k), kw, k)

        print(f"Recall@{k}:")
        print(f"  纯向量         : {only_vec}/{n} = {only_vec/n:.1%}")
        print(f"  向量+BM25      : {hybrid}/{n} = {hybrid/n:.1%}")
        print(f"  +Reranker      : {hybrid_rerank}/{n} = {hybrid_rerank/n:.1%}\n")


if __name__ == "__main__":
    main()
