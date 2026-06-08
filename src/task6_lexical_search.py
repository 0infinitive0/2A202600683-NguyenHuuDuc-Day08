"""
Task 6 — Lexical Search Module.

Viết module thực hiện lexical search. Mặc định sử dụng BM25.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Sử dụng thuật toán BM25 (thư viện rank_bm25)

Cài đặt bổ sung:
    pip install rank-bm25
"""

from pathlib import Path
import re
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Cấu hình Chunking đồng bộ với Task 4 để đảm bảo tính nhất quán dữ liệu
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def tokenize(text: str) -> list[str]:
    """
    Hàm tokenize đơn giản: Chuyển về chữ thường và tách từ bằng regex.
    Có thể nâng cấp bằng Underthesea nếu muốn tối ưu tách từ tiếng Việt.
    """
    text = text.lower()
    # Tách chuỗi theo các ký tự không phải chữ cái hoặc số
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def load_and_chunk_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/ và tiến hành chunking
    tương tự Task 4 để làm cơ sở dữ liệu nền cho BM25.
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"Thư mục {STANDARDIZED_DIR} không tồn tại!")
        return documents

    # 1. Load Documents
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file.parent) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })

    # 2. Chunking Documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "doc_type": doc["metadata"]["type"],
                    "chunk_index": i
                }
            })
    return chunks


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng thuật toán BM25.

    Args:
        query: Câu truy vấn từ khóa
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # BM25 score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Bước 1: Chuẩn bị danh sách các chunk dữ liệu
    chunks = load_and_chunk_documents()
    if not chunks:
        print("Không có dữ liệu văn bản để thực hiện tìm kiếm từ khóa.")
        return []

    # Bước 2: Tokenize toàn bộ văn bản để đưa vào mô hình BM25
    corpus_tokens = [tokenize(chunk["content"]) for chunk in chunks]
    
    # Bước 3: Khởi tạo mô hình BM25Okapi với tập ngữ liệu đã tokenize
    bm25 = BM25Okapi(corpus_tokens)
    
    # Bước 4: Tokenize câu truy vấn (query) và tính toán điểm số BM25
    query_tokens = tokenize(query)
    doc_scores = bm25.get_scores(query_tokens)
    
    # Bước 5: Tổ chức cấu trúc kết quả trả về bao gồm nội dung, score và metadata
    search_results = []
    for idx, score in enumerate(doc_scores):
        # Chỉ lấy các đoạn có score > 0 (tức là có chứa ít nhất một từ khóa trong câu truy vấn)
        if score > 0:
            search_results.append({
                "content": chunks[idx]["content"],
                "score": float(score),
                "metadata": chunks[idx]["metadata"]
            })
            
    # Bước 6: Sắp xếp kết quả theo điểm số BM25 giảm dần (score descending)
    search_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Trả về kết quả giới hạn bởi top_k
    return search_results[:top_k]


if __name__ == "__main__":
    # Chạy thử nghiệm Module Lexical Search (BM25)
    print("=" * 50)
    print("Testing Task 6: Lexical Search (BM25)")
    print("=" * 50)
    
    query_test = "hình phạt cho tội tàng trữ ma tuý"
    print(f"Query: '{query_test}'\n")
    
    try:
        results = lexical_search(query_test, top_k=5)
        if not results:
            print("Không tìm thấy kết quả từ khóa nào phù hợp hoặc chưa có dữ liệu tại data/standardized/.")
        else:
            for i, r in enumerate(results, 1):
                print(f"{i}. [BM25 Score: {r['score']:.3f}] - Nguồn: {r['metadata']['source']}")
                print(f"   Nội dung: {r['content'][:150]}...")
                print("-" * 40)
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình tìm kiếm từ khóa: {e}")