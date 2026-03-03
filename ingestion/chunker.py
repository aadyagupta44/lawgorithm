"""
Text chunking and splitting for repository files.

This module implements a `DocumentChunker` class which leverages
LangChain's `RecursiveCharacterTextSplitter` to break long documents
into smaller overlapping chunks suitable for embedding and retrieval.
It exposes a `chunk_documents` method that converts file dictionaries
into a list of chunk dictionaries enriched with metadata.
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunker:
	"""Chunk documents into smaller text pieces for embedding.

	The chunker is configured with `CHUNK_SIZE` and `CHUNK_OVERLAP`
	from the project's config and provides a single method
	`chunk_documents` that accepts the file dictionaries returned
	by `GitHubLoader` and outputs chunk dictionaries ready for
	embedding and storage.
	"""

	def __init__(self) -> None:
		"""Initialize the RecursiveCharacterTextSplitter.

		The constructor creates a `RecursiveCharacterTextSplitter` using
		configured chunk size and overlap. It prints a confirmation so
		the developer knows the chunker is ready.
		"""

		# create the text splitter instance with configured sizes
		self.splitter: RecursiveCharacterTextSplitter = (
			RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
		)

		# notify the user that the chunker was initialized
		print(f"[INFO] DocumentChunker ready (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})")

	def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, object]]:
		"""Split page-level inputs into chunks with document provenance.

		Parameters:
			pages: A list of page dictionaries produced by `DocumentLoader`.
			Each page dict must contain: `content`, `filename`, `file_type`,
			`page_number`, `total_pages`, and `document_id`.

		Returns:
			A list of chunk dictionaries. Each chunk dictionary contains:
			`content`, `filename`, `file_type`, `page_number`, `chunk_index`,
			`document_id`, and `total_pages` for downstream provenance.
		"""

		chunks: List[Dict[str, object]] = []  # accumulator for all produced chunks
		total_created = 0  # counter for reporting

		# iterate over provided pages and split their content
		for page in documents:
			# defensive checks: skip if no content field or empty
			if not page or "content" not in page:
				print(f"[SKIP] Page missing content or malformed: {page.get('filename', 'unknown') if isinstance(page, dict) else 'invalid'}")
				continue

			content = page.get("content", "")  # extract content safely
			if not content or not content.strip():
				# skip empty pages to avoid creating empty chunks
				print(f"[SKIP] Empty content for page: {page.get('filename', 'unknown')}")
				continue

			# use the splitter to break content into text chunks
			try:
				text_chunks = self.splitter.split_text(content)
			except Exception as e:
				# handle splitting errors gracefully and continue
				print(f"[ERROR] Failed to split page for '{page.get('filename', 'unknown')}': {e}")
				continue

			# extract provenance metadata from the page
			filename = page.get("filename", "")
			file_type = page.get("file_type", "")
			page_number = page.get("page_number", 1)
			document_id = page.get("document_id")
			total_pages = page.get("total_pages")

			# create chunk dictionaries with metadata for each produced chunk
			for idx, chunk_text in enumerate(text_chunks):
				chunk_dict: Dict[str, object] = {
					"content": chunk_text,  # the textual content of this chunk
					"filename": filename,  # original filename
					"file_type": file_type,  # original file type (pdf/text)
					"page_number": page_number,  # 1-based source page number
					"chunk_index": idx,  # zero-based index of this chunk within the page
					"document_id": document_id,  # stable id for the original document
					"total_pages": total_pages,  # total pages in source document
				}
				chunks.append(chunk_dict)  # add to accumulator
				total_created += 1

		# print summary of chunking operation
		print(f"[INFO] Chunking complete. Total chunks created: {total_created}")
		return chunks

