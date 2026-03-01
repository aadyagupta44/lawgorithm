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
		"""Split each document into chunks and return chunk metadata.

		Parameters:
			documents: A list of file dictionaries with keys `content`,
					   `filename`, `filepath`, `file_type`, and `repo_name`.

		Returns:
			A list of chunk dictionaries. Each chunk dictionary contains
			`content`, `filename`, `filepath`, `file_type`, `repo_name`,
			and `chunk_index` indicating the chunk's position.
		"""

		chunks: List[Dict[str, object]] = []  # accumulator for all produced chunks
		total_created = 0  # counter for reporting

		# iterate over provided documents and split their content
		for doc in documents:
			# defensive checks: skip if no content field or empty
			if not doc or "content" not in doc:
				print(f"[SKIP] Document missing content or malformed: {doc.get('filepath', 'unknown') if isinstance(doc, dict) else 'invalid'}")
				continue

			content = doc.get("content", "")  # extract content safely
			if not content or not content.strip():
				# skip empty documents to avoid creating empty chunks
				print(f"[SKIP] Empty content for file: {doc.get('filepath', 'unknown')}")
				continue

			# use the splitter to break content into text chunks
			try:
				text_chunks = self.splitter.split_text(content)
			except Exception as e:
				# handle splitting errors gracefully and continue
				print(f"[ERROR] Failed to split document '{doc.get('filepath', 'unknown')}': {e}")
				continue

			# create chunk dictionaries with metadata for each produced chunk
			for idx, chunk_text in enumerate(text_chunks):
				chunk_dict: Dict[str, object] = {
					"content": chunk_text,  # the textual content of this chunk
					"filename": doc.get("filename", ""),  # original filename
					"filepath": doc.get("filepath", ""),  # original file path in repo
					"file_type": doc.get("file_type", ""),  # original file extension
					"repo_name": doc.get("repo_name", ""),  # originating repository name
					"chunk_index": idx,  # zero-based index of this chunk within the document
				}
				chunks.append(chunk_dict)  # add to accumulator
				total_created += 1

		# print summary of chunking operation
		print(f"[INFO] Chunking complete. Total chunks created: {total_created}")
		return chunks

