---

# 📝 `doc_to_md.py`: Universal Docs → Markdown Dumper

This is a single-file, no-dependency CLI utility (besides `crawl4ai`) that crawls documentation websites and dumps their cleaned content as Markdown files.
It works out-of-the-box and saves progress to avoid redundant re-crawling.
No config files, no setup – just run it and point it to a docs site.

---

## 🚀 What It Does

* Starts from a given documentation URL.
* Recursively crawls all internal pages under the same domain and path prefix.
* Filters out non-text resources (e.g. images, videos, binaries).
* Extracts main content areas (using default tag/selector heuristics).
* Converts HTML to clean Markdown using `crawl4ai`'s pipeline.
* Saves each page as a `.md` file with a YAML frontmatter header.
* Also saves the raw cleaned HTML in a parallel `html_dump/` folder.
* Avoids duplicate saves by default, using content hashing.
* Marks repeated paragraphs across docs as “repeating chunks”.

---

## 📦 Output Structure

```
your-docs-folder/
├── some-page.md
├── another-section.md
├── html_dump/
│   ├── some-page.html
│   └── another-section.html
└── _manifest.json
```

* Markdown files with clean content and metadata.
* `html_dump/` for raw HTML backups.
* `_manifest.json` summarizing crawl results.

---

## 🛠️ Usage

```bash
python doc_to_md.py https://example.com/docs/
```

* Prompts for a URL if none provided.
* Saves output in a folder named after the domain and top-level path (e.g., `example.com_docs`).

### Optional Flags

* `--test` → Crawls only the starting page (no recursion).
* `--no-unique` → Disables deduplication (saves all pages, even identical ones).
* `--unique` → Forces deduplication on (default).

---

## 🔄 Resuming Crawls

Interrupted runs save progress in `progress.json`. On restart:

* Resumes from where it left off.
* Automatically skips seen or saved pages.

When complete, `progress.json` is deleted.

---

## 🧠 How It Works (High Level)

* URL normalization to avoid redundant fetches.
* HTML parsing with `lxml` to extract canonical links and internal anchors.
* `crawl4ai` handles JavaScript rendering, content scraping, and Markdown conversion.
* All crawled content is hashed and indexed for duplication detection.
* Progress saved incrementally, supports clean shutdown via `SIGINT` or `SIGTERM`.

---

## 🧾 Dependencies

Only one external package is required (and assumes it’s installed):

* [`crawl4ai`](https://pypi.org/project/crawl4ai/)

```bash
pip install crawl4ai
```

---

## 🧹 Cleanup

If you abort midway or crash, rerun with the same URL – it’ll pick up from saved progress.
When the crawl finishes cleanly, `progress.json` is removed.

---

## 💬 Why Use This?

* Need a local Markdown archive of a documentation site. (best for RAG retrieval or creating a knowledge base)
* Archiving or syncing technical docs offline.
* Works without setting up a whole crawler framework – just one script.

---

## 🧯 Notes

* Binary files like `.pdf`, `.zip`, `.png`, etc. are ignored.
* Cookie banners and feedback widgets are filtered out automatically.
* Script is tuned for docs, not general-purpose scraping.

---
