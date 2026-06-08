"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: Số lượng 5 chunk (~1000-2000 tokens) đủ để cung cấp evidence đa dạng 
# mà không làm tăng quá nhiều kích thước context dẫn đến hiện tượng lost in the middle
# (LLM "bỏ quên" thông tin ở giữa).
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: Giá trị 0.9 giúp câu trả lời tự nhiên và đa dạng về cách hành văn,
# nhưng vẫn giữ được sự kiểm soát để không sinh ra các hallucination quá xa rời
# context thực tế.
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: Trong hệ thống RAG yêu cầu citation, câu trả lời cần phải có
# độ chính xác cao và dựa vào sự kiện có thật (factual). Temperature thấp (0.3)
# giúp giảm thiểu tính sáng tạo và bay bổng của model.
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'I cannot verify this information' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation in the format [Source, Year] or similar explicit reference.
- If context is insufficient or there's not enough evidence, return exactly 'I cannot verify this information'.
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks
    
    reordered = []
    # Các item ở index chẵn (0, 2, 4...) đặt ở đầu (theo thứ tự: tốt nhất trước)
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    # Các item ở index lẻ (1, 3, 5...) đặt ở cuối, theo thứ tự ngược lại (tốt nhì ở cuối cùng)
    for i in range(len(chunks) - 1 - (len(chunks) % 2 == 0), 0, -2):
        reordered.append(chunks[i])
        
    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", metadata.get("filename", f"Source {i}"))
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)
    
    if not chunks:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none"
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)
    
    # Step 3: Format context
    context = format_context(reordered)
    
    # Step 4: Build prompt
    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""
    
    # Step 5: Call LLM
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model=os.getenv("CUSTOM_LLM_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("CUSTOM_LLM_KEY"),
            openai_api_base=os.getenv("CUSTOM_LLM_URL"),
            temperature=TEMPERATURE,
            model_kwargs={"top_p": TOP_P}
        )
        
        response = llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("user", user_message)
        ])
        
        answer = response.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        answer = "I cannot verify this information"

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
