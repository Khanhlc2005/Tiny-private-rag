"""
CLI tìm kiếm để kiểm tra thủ công retrieval.
"""
import argparse
import sys

from app.retrieval.hybrid import HybridRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Tìm chunk liên quan trong Qdrant")
    parser.add_argument("question", help="Câu hỏi cần tìm")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--source", default=None, help="Chỉ tìm trong file này")
    parser.add_argument("--dense-only", action="store_true", help="Bỏ nhánh sparse")
    args = parser.parse_args()

    retriever = HybridRetriever()
    try:
        if args.dense_only:
            results = retriever.search_dense_only(
                args.question, top_k=args.top_k, source_file=args.source
            )
        else:
            results = retriever.search(
                args.question, top_k=args.top_k, source_file=args.source
            )
    except RuntimeError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1

    mode = "dense-only" if args.dense_only else "hybrid (RRF)"
    print(f"Chế độ: {mode} | {len(results)} kết quả")
    for rank, r in enumerate(results, start=1):
        preview = " ".join(r.text.split())[:200]
        print(f"\n#{rank} score={r.score:.4f} | {r.source_file} | trang {r.page} | {r.chunk_id}")
        print(f"   {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())