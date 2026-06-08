"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown
import re
import html as _html

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.rglob("*"):
        if not filepath.is_file():
            continue

        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        rel_path = filepath.relative_to(legal_dir)
        output_path = output_dir / rel_path.with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Converting: {filepath}")
        try:
            result = md.convert(str(filepath))
            # MarkItDown result usually exposes .text_content
            content = getattr(result, "text_content", None) or getattr(result, "content", None) or str(result)
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")
        except Exception as exc:
            print(f"  ✗ Failed to convert {filepath}: {exc}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.rglob("*"):
        if not filepath.is_file():
            continue

        if filepath.suffix.lower() != ".json":
            continue

        rel_path = filepath.relative_to(news_dir)
        output_path = output_dir / rel_path.with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Converting: {filepath}")
        try:
            raw = filepath.read_text(encoding="utf-8")
            data = json.loads(raw)

            # Normalize fields from multiple possible schemas
            meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}

            title = (
                data.get("title")
                or meta.get("news_title")
                or meta.get("title")
                or data.get("news_title")
                or "Unknown"
            )

            url = (
                data.get("url")
                or meta.get("source_url")
                or meta.get("url")
                or data.get("source_url")
                or "N/A"
            )

            date_crawled = (
                data.get("date_crawled")
                or meta.get("crawl_date")
                or meta.get("date_crawled")
                or data.get("crawl_date")
                or "N/A"
            )

            # Prefer already-extracted markdown text
            content_markdown = data.get("content_markdown") or data.get("content")

            # If no markdown, try cleaned_html/fit_html/html fields and create text and html entries
            content_html = None
            content_text = None

            if not content_markdown:
                content_html = data.get("cleaned_html") or data.get("fit_html") or data.get("html")

                if isinstance(content_html, str) and content_html.strip():
                    # Create a plain-text fallback by stripping tags
                    text = re.sub(r"<script[\s\S]*?</script>", "", content_html, flags=re.I)
                    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
                    text = re.sub(r"<[^>]+>", "", text)
                    text = _html.unescape(text)
                    # Normalize whitespace
                    content_text = re.sub(r"\s+", " ", text).strip()

            # Build standardized JSON object
            standardized = {
                "title": title,
                "url": url,
                "date_crawled": date_crawled,
                # keep original raw metadata for reference
                "raw_metadata": meta or data.get("metadata") or {},
            }

            if content_markdown:
                standardized["content_markdown"] = content_markdown
            if content_html:
                standardized["content_html"] = content_html
            if content_text:
                standardized["content_text"] = content_text

            # Save standardized JSON (preserve folder structure)
            output_path.write_text(json.dumps(standardized, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ Saved JSON: {output_path}")
        except Exception as exc:
            print(f"  ✗ Failed to convert {filepath}: {exc}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
