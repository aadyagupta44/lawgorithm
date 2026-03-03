"""
Embedding and Pinecone vector database storage.

This module implements a `PineconeEmbedder` class which is responsible for
converting text chunks into embedding vectors using a HuggingFace
sentence-transformers model and storing those vectors in a Pinecone
index. The class handles index initialization, batched upserts, and
namespace generation so multiple repositories can be stored safely.
"""

from typing import List, Dict
import time  # used for simple rate limiting between batches
import re  # used to sanitize namespace strings

from sentence_transformers import SentenceTransformer  # embedding model
from pinecone import Pinecone  # new Pinecone client style

from config import (
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION
)


class PineconeEmbedder:
    """Embed text chunks and store them in Pinecone.

    This class loads a HuggingFace SentenceTransformer model to compute
    dense vector embeddings for chunk texts and upserts them into a
    Pinecone index. It expects chunk dictionaries produced by
    `DocumentChunker` which include `document_id` and `page_number` so
    that vector metadata preserves document provenance.
    """

    def __init__(self) -> None:
        """Initialize embeddings model and Pinecone client."""

        # load the sentence-transformers model for computing embeddings
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"[OK] Embedding model '{EMBEDDING_MODEL}' loaded successfully")
        except Exception as e:
            self.model = None
            print(f"[ERROR] Failed to load embedding model '{EMBEDDING_MODEL}': {e}")

        # initialize Pinecone using the new Pinecone class style
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)

            # get list of existing index names
            existing_indexes = [index.name for index in pc.list_indexes()]

            # connect to our codemind index
            if PINECONE_INDEX_NAME in existing_indexes:
                self.index = pc.Index(PINECONE_INDEX_NAME)
                print(f"[OK] Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
            else:
                # index doesn't exist — user needs to create it in dashboard
                self.index = None
                print(f"[ERROR] Index '{PINECONE_INDEX_NAME}' not found in Pinecone")
                print("[INFO] Please create the index in your Pinecone dashboard first")

        except Exception as e:
            self.index = None
            print(f"[ERROR] Failed to initialize Pinecone client: {e}")

        print(f"[INFO] PineconeEmbedder initialized (index={PINECONE_INDEX_NAME})")

    def embed_and_store(
        self,
        chunks: List[Dict[str, object]],
        namespace: str
    ) -> int:
        """Embed provided chunks and store them in Pinecone.

        Parameters:
            chunks: List of chunk dictionaries containing content and metadata
            namespace: Namespace string to isolate vectors per repository

        Returns:
            Total number of vectors successfully stored
        """

        # guard against empty input
        if not chunks:
            print("[WARN] No chunks provided to embed_and_store")
            return 0

        # guard against uninitialized model or index
        if self.model is None:
            print("[ERROR] Embedding model not loaded")
            return 0

        if self.index is None:
            print("[ERROR] Pinecone index not available")
            return 0

        total_upserted = 0  # track how many vectors we store
        batch_size = 100    # process 100 chunks at a time

        # helper to create a unique id for each chunk using document id
        def _make_id(ns: str, document_id: str, idx: int) -> str:
            # sanitize document id and namespace
            safe_doc = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", str(document_id))
            safe_ns = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", ns)
            return f"{safe_ns}:{safe_doc}:{idx}"

        # process chunks in batches
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start: start + batch_size]

            # extract text content from each chunk for embedding
            texts = [str(c.get("content", "")) for c in batch]

            # compute embeddings for this batch
            try:
                vectors = self.model.encode(
                    texts,
                    show_progress_bar=False
                ).tolist()
            except Exception as e:
                print(f"[ERROR] Failed to compute embeddings for batch at {start}: {e}")
                continue

            # build list of (id, vector, metadata) tuples for Pinecone
            upsert_items = []
            for i, vec in enumerate(vectors):
                chunk = batch[i]
                document_id = str(chunk.get("document_id", ""))
                chunk_index = int(chunk.get("chunk_index", 0))
                vid = _make_id(namespace, document_id, chunk_index)

                # metadata stored alongside each vector in Pinecone
                metadata = {
                    "filename": str(chunk.get("filename", "")),
                    "file_type": str(chunk.get("file_type", "")),
                    "page_number": int(chunk.get("page_number", 0)),
                    "chunk_index": chunk_index,
                    "document_id": document_id,
                    "total_pages": int(chunk.get("total_pages", 0)) if chunk.get("total_pages") is not None else None,
                    "content_preview": str(chunk.get("content", ""))[:1000],
                }
                upsert_items.append((vid, vec, metadata))

            # upsert this batch into Pinecone
            try:
                self.index.upsert(
                    vectors=upsert_items,
                    namespace=namespace
                )
                total_upserted += len(upsert_items)
                print(f"[OK] Upserted batch of {len(upsert_items)} vectors "
                      f"(total: {total_upserted})")
            except Exception as e:
                print(f"[ERROR] Failed to upsert batch: {e}")

            # small pause to be respectful to the API
            time.sleep(0.1)

        print(f"[INFO] Embedding complete. Total vectors stored: {total_upserted}")
        return total_upserted

    def namespace_from_filename(self, filename: str) -> str:
        """Create a Pinecone namespace string from a filename.

        Parameters:
            filename: The source filename (e.g. 'contract.pdf')

        Returns:
            A sanitized namespace string suitable for Pinecone.
        """

        try:
            cleaned = str(filename or "").strip()
            # remove path separators and extensions
            cleaned = re.sub(r"\.[a-zA-Z0-9]+$", "", cleaned)
            cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", cleaned)
            return cleaned or "default_namespace"
        except Exception as e:
            print(f"[WARN] Could not create namespace from filename: {e}")
            return "default_namespace"
