"""
DocumentLoader for Lawgorithm ingestion pipeline.

This module provides a `DocumentLoader` class to read legal documents
from disk (PDFs) or raw text, normalize them into a simple page-level
dictionary format, and return lists of page dictionaries ready for
chunking and embedding.

The loader handles encrypted PDFs gracefully and prints progress so
developers can see ingestion status during long runs.
"""

from typing import List, Dict
import os  # filesystem utilities
import uuid  # used to create unique document IDs
import fitz  # PyMuPDF for PDF parsing


class DocumentLoader:
    """Load documents (PDF/text) and return page-level dictionaries.

    Each returned page dict contains: content, page_number, filename,
    file_type, total_pages, document_id. The same document_id is used
    for all pages belonging to a single document.
    """

    def __init__(self) -> None:
        """Initialize the loader and print confirmation."""
        # notify that the loader is available
        print("[INFO] DocumentLoader initialized")

    def load_pdf(self, file_path: str) -> List[Dict[str, object]]:
        """Extract text from a PDF file page-by-page.

        Parameters:
            file_path: Path to the PDF file on local disk.

        Returns:
            A list of dictionaries, one per page, with keys:
            content, page_number, filename, file_type, total_pages, document_id.
        """

        pages: List[Dict[str, object]] = []  # accumulator for page dicts

        # create a stable document id for this file
        document_id = uuid.uuid4().hex  # unique id for the document

        # safe filename extraction for metadata
        filename = os.path.basename(file_path)  # filename with extension

        try:
            # open the document using PyMuPDF
            doc = fitz.open(file_path)
        except Exception as e:
            # if the file cannot be opened, log and return empty list
            print(f"[ERROR] Could not open PDF '{file_path}': {e}")
            return pages

        # handle encrypted PDFs by attempting to authenticate with empty password
        try:
            if doc.is_encrypted:
                try:
                    # try to authenticate with an empty password (common case)
                    doc.authenticate("")
                except Exception:
                    # if authentication fails, report and skip the file
                    print(f"[WARN] PDF '{file_path}' is encrypted and cannot be read")
                    doc.close()
                    return pages
        except Exception:
            # ignore any property access errors and continue
            pass

        # get total pages for metadata
        try:
            total_pages = doc.page_count
        except Exception:
            total_pages = 0

        # iterate through pages and extract text
        for page_number in range(total_pages):
            try:
                page = doc.load_page(page_number)  # zero-based
                text = page.get_text("text")  # extract plain text
            except Exception as e:
                # continue on per-page errors but report them
                print(f"[WARN] Failed to extract page {page_number+1} of '{filename}': {e}")
                continue

            # assemble page dictionary
            page_dict: Dict[str, object] = {
                "content": text,  # extracted text of the page
                "page_number": page_number + 1,  # 1-based page number
                "filename": filename,  # original filename
                "file_type": "pdf",  # file type indicator
                "total_pages": total_pages,  # total number of pages in the doc
                "document_id": document_id,  # unique id for the document
            }

            pages.append(page_dict)  # add to result list
            print(f"[OK] Extracted page {page_number+1}/{total_pages} from {filename}")

        # close document file handle
        try:
            doc.close()
        except Exception:
            pass

        print(f"[INFO] Completed PDF load: {filename} ({len(pages)} pages)")
        return pages

    def load_text(self, text: str, filename: str) -> List[Dict[str, object]]:
        """Wrap raw text into the same page-level dict format.

        Parameters:
            text: Raw document text.
            filename: Logical filename to attach to the document.

        Returns:
            A list containing a single page dictionary representing the text.
        """

        document_id = uuid.uuid4().hex  # unique id for this text document

        page_dict: Dict[str, object] = {
            "content": text,  # full text content
            "page_number": 1,  # single-page document
            "filename": filename,  # provided filename
            "file_type": "text",  # text type indicator
            "total_pages": 1,  # only one page
            "document_id": document_id,  # unique id
        }

        print(f"[INFO] Loaded text document: {filename}")
        return [page_dict]

    def load_multiple(self, sources: List[Dict[str, str]]) -> List[Dict[str, object]]:
        """Load multiple sources (PDFs or text) and combine into page list.

        Parameters:
            sources: List of source dictionaries with keys:
                     type: 'pdf' or 'text'
                     content: file path for pdf or raw text for text
                     filename: name to use in metadata

        Returns:
            Combined list of page-level dictionaries for all sources.
        """

        all_pages: List[Dict[str, object]] = []  # accumulator across sources

        for src in sources:
            src_type = src.get("type")  # expected 'pdf' or 'text'
            content = src.get("content")  # path or raw text
            filename = src.get("filename", "unknown")  # fallback filename

            if src_type == "pdf":
                # call PDF loader and extend results
                pages = self.load_pdf(content)
                all_pages.extend(pages)
            elif src_type == "text":
                # call text loader
                pages = self.load_text(content or "", filename)
                all_pages.extend(pages)
            else:
                # unknown type; warn and skip
                print(f"[WARN] Unknown source type '{src_type}' for filename '{filename}' — skipping")
                continue

        print(f"[INFO] Loaded total pages from sources: {len(all_pages)}")
        return all_pages
