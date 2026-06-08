"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Configure, Property, DataType

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Lựa chọn phương pháp cắt văn bản: "recursive" (RecursiveCharacterTextSplitter)
# - Lý do chọn Recursive: Đây là phương pháp an toàn và phổ biến nhất, giúp giữ các đoạn văn, 
#   câu có nghĩa đi liền với nhau bằng cách thử nghiệm cắt theo thứ tự ưu tiên ký tự (\n\n, \n, khoảng trắng).
CHUNKING_METHOD = "recursive"  

# Lựa chọn kích thước Chunk Size = 500 ký tự
# - Lý do: Phù hợp với các văn bản pháp luật ngắn, các điều khoản cụ thể hoặc các đoạn tin tức ngắn. 
#   Đủ nhỏ để mô hình embedding tập trung vào ngữ nghĩa hẹp của đoạn, tránh bị nhiễu thông tin.
CHUNK_SIZE = 500        

# Lựa chọn độ gối đầu Chunk Overlap = 50 ký tự
# - Lý do: Chiếm khoảng 10% kích thước chunk. Giúp đảm bảo ngữ cảnh ở ranh giới giữa 2 chunk 
#   không bị mất mát, duy trì tính liên tục của thông tin khi thực hiện tìm kiếm ngữ nghĩa.
CHUNK_OVERLAP = 50      

# Lựa chọn mô hình Embedding: "sentence-transformers/all-MiniLM-L6-v2"
# - Lý do: Đây là mô hình cực kỳ nhẹ (lightweight), tốc độ xử lý nhanh, tiết kiệm tài nguyên 
#   và chạy mượt mà ở môi trường local mà không cần phần cứng quá mạnh.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
EMBEDDING_DIM = 384  # Kích thước vector đặc trưng đầu ra của mô hình all-MiniLM-L6-v2

# Lựa chọn Vector Store: "weaviate"
# - Lý do: Weaviate hỗ trợ hybrid search mạnh mẽ (kết hợp cả vector search và từ khóa BM25), 
#   quản lý schema chặt chẽ và xử lý batch insert tối ưu cho các hệ thống RAG.
VECTOR_STORE = "weaviate"  


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"Thư mục {STANDARDIZED_DIR} không tồn tại!")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # Phân loại tài liệu dựa trên đường dẫn thư mục cha
        doc_type = "legal" if "legal" in str(md_file.parent) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn (RecursiveCharacterTextSplitter).

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
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


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng mô hình sentence-transformers đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    
    # Tiến hành tính toán embedding vector cho toàn bộ văn bản
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks và các vector tương ứng vào Weaviate Vector Store sử dụng Client v4.
    """
    # Kết nối tới instance Weaviate local (mặc định chạy tại http://localhost:8080)
    with weaviate.connect_to_local() as client:
        collection_name = "DrugLawDocs"
        
        # Xóa collection cũ nếu đã tồn tại để tránh trùng lặp dữ liệu khi re-run
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            
        # Tạo cấu trúc Collection mới với cấu hình Vector thủ công (none) vì ta tự feed vector từ ngoài vào
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )
        
        # Sử dụng cơ chế dynamic batching để nạp dữ liệu vào Weaviate một cách tối ưu nhất
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["doc_type"],
                        "chunk_index": chunk["metadata"]["chunk_index"]
                    },
                    vector=chunk["embedding"]
                )
                
        # Kiểm tra nhanh số lượng record đã được insert thành công
        obj_count = len(collection.query.fetch_objects(limit=1).objects)
        if obj_count > 0:
            print(f"✓ Đã nạp thành công các chunks vào collection '{collection_name}' trong Weaviate.")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        print("Không tìm thấy tài liệu nào để xử lý. Vui lòng kiểm tra lại data/standardized/")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store successfully!")


if __name__ == "__main__":
    run_pipeline()