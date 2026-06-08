import argparse
import argparse
import asyncio
import sys
import warnings
import os
import re
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

from crawl4ai import AsyncWebCrawler


def _slugify(s: str, maxlen: int = 80) -> str:
    s = s or ""
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    s = s.strip("-")
    return s[:maxlen] or str(uuid.uuid4())


def _extract_html(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "get"):
        for key in ("html", "content", "text", "body"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)
    return str(result)


def _extract_title(result_html: str, result) -> Optional[str]:
    if hasattr(result, "get"):
        t = result.get("title")
        if isinstance(t, str) and t.strip():
            return t.strip()
    if isinstance(result_html, str):
        m = re.search(r"<title[^>]*>(.*?)</title>", result_html, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
        m = re.search(r"<h1[^>]*>(.*?)</h1>", result_html, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return None


async def crawl_and_save(
    url: str,
    output_dir: str = "data/landing/news",
    file_format: str = "json",
) -> Tuple[str, dict]:
    os.makedirs(output_dir, exist_ok=True)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    html = _extract_html(result)
    title = _extract_title(html, result) or ""
    crawl_date = datetime.now(timezone.utc).isoformat()
    metadata = {
        "source_url": url,
        "crawl_date": crawl_date,
        "news_title": title,
    }

    base = _slugify(title) if title else _slugify(f"{url}-{crawl_date}-{uuid.uuid4().hex[:8]}")
    if file_format.lower() == "html":
        filename = f"{base}.html"
        path = os.path.join(output_dir, filename)
        meta_comment = "<!--\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n-->\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(meta_comment)
            f.write(html)
    else:
        filename = f"{base}.json"
        path = os.path.join(output_dir, filename)
        payload = {"metadata": metadata, "html": html}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return path, metadata


def _save_result(result, url: str, output_dir: str = "data/landing/news", file_format: str = "json") -> Tuple[str, dict]:
    """Save a crawled result (no network) to disk and return (path, metadata)."""
    os.makedirs(output_dir, exist_ok=True)

    html = _extract_html(result)
    title = _extract_title(html, result) or ""
    crawl_date = datetime.now(timezone.utc).isoformat()
    metadata = {"source_url": url, "crawl_date": crawl_date, "news_title": title}

    base = _slugify(title) if title else _slugify(f"{url}-{crawl_date}-{uuid.uuid4().hex[:8]}")
    if file_format.lower() == "html":
        filename = f"{base}.html"
        path = os.path.join(output_dir, filename)
        meta_comment = "<!--\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n-->\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(meta_comment)
            f.write(html)
    else:
        filename = f"{base}.json"
        path = os.path.join(output_dir, filename)
        payload = {"metadata": metadata, "html": html}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return path, metadata


async def _run(urls, output_dir, file_format, quiet):
    results = []
    # Use a single AsyncWebCrawler for all URLs to avoid creating many transports
    # which can trigger resource warnings during interpreter shutdown on Windows.
    try:
        async with AsyncWebCrawler() as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(url=url)
                    path, metadata = _save_result(result, url=url, output_dir=output_dir, file_format=file_format)
                    results.append((path, metadata))
                    if not quiet:
                        try:
                            print(path)
                            print(json.dumps(metadata, ensure_ascii=False))
                        except Exception:
                            # Ignore print errors during shutdown
                            pass
                except Exception as e:
                    print(f"Error crawling {url}: {e}")
    except Exception as e:
        print(f"Crawler error: {e}")

    return results


def main():
    # Ensure the event loop policy supports subprocesses on Windows. A
    # SelectorEventLoopPolicy on Windows does not implement subprocess support
    # (create_subprocess_exec) and raises NotImplementedError. Use the
    # Proactor policy if the current policy is the selector one.
    if sys.platform.startswith("win"):
        policy = asyncio.get_event_loop_policy()
        if policy.__class__.__name__ == "WindowsSelectorEventLoopPolicy":
            try:
                from asyncio import WindowsProactorEventLoopPolicy

                asyncio.set_event_loop_policy(WindowsProactorEventLoopPolicy())
            except Exception:
                # If Proactor policy isn't available, continue and let subprocess
                # creation raise a clear error later.
                pass

    # Suppress ResourceWarning messages about unclosed transports that can
    # happen during shutdown when file descriptors are already closed.
    warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport.*")

    parser = argparse.ArgumentParser(description="Crawl URL(s) and save each news item to files")
    parser.add_argument("urls", nargs="+", help="One or more URLs to crawl")
    parser.add_argument("--output-dir", default="data/landing/news", help="Directory to save files")
    parser.add_argument("--format", choices=["json", "html"], default="json", help="File format to save")
    parser.add_argument("--quiet", action="store_true", help="Only print file paths")
    args = parser.parse_args()

    asyncio.run(_run(args.urls, args.output_dir, args.format, args.quiet))


if __name__ == "__main__":
    main()
