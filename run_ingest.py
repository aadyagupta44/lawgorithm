"""
End-to-end ingestion test runner for CodeMind.

This file tests the complete ingestion pipeline by:
1. Fetching files from a real GitHub repository
2. Chunking those files into smaller pieces
3. Embedding and storing them in Pinecone

This file can be deleted before final deployment.

Usage: python run_ingest.py
"""

from ingestion import GitHubLoader, DocumentChunker, PineconeEmbedder


# Using flask because it is small, clean and well documented
# Perfect for testing without waiting too long
REPO_URL = "https://github.com/realpython/flask-boilerplate"

def main() -> None:
    """Run the full ingestion pipeline end to end."""

    print("\n========================================")
    print("   CodeMind Ingestion Pipeline Test     ")
    print("========================================\n")

    # ---- Step 1: Fetch files from GitHub ----
    print("STEP 1: Loading files from GitHub...")
    print(f"Repository: {REPO_URL}\n")

    # initialize the loader with our target repo URL
    loader = GitHubLoader(REPO_URL)

    # fetch all relevant files from the repository
    files = loader.get_all_files()

    # print summary of what was fetched
    print(f"\n✓ Fetched {len(files)} files from GitHub\n")

    # guard: if no files were fetched something went wrong
    if len(files) == 0:
        print("[ERROR] No files were fetched. Check your GITHUB_TOKEN in .env")
        return

    # ---- Step 2: Chunk the documents ----
    print("STEP 2: Chunking documents into smaller pieces...")

    # initialize the chunker with settings from config
    chunker = DocumentChunker()

    # split all fetched files into chunks
    chunks = chunker.chunk_documents(files)

    # print summary of chunking
    print(f"\n✓ Created {len(chunks)} chunks from {len(files)} files\n")

    # guard: if no chunks were created something went wrong
    if len(chunks) == 0:
        print("[ERROR] No chunks were created. Check your files have content.")
        return

    # ---- Step 3: Embed and store in Pinecone ----
    print("STEP 3: Embedding chunks and storing in Pinecone...")
    print("This may take several minutes depending on number of chunks...\n")

    # initialize the embedder which connects to Pinecone
    embedder = PineconeEmbedder()

    # generate a safe namespace from the repo URL
    # this keeps different repos separated in Pinecone
    namespace = embedder.namespace_from_url(REPO_URL)
    print(f"Using Pinecone namespace: '{namespace}'\n")

    # embed all chunks and store them in Pinecone
    stored = embedder.embed_and_store(chunks, namespace)

    # print final summary
    print(f"\n✓ Stored {stored} vectors in Pinecone under namespace '{namespace}'\n")

    # ---- Final Summary ----
    print("========================================")
    print("   Ingestion Pipeline Complete!         ")
    print("========================================")
    print(f"  Repository : {REPO_URL}")
    print(f"  Files      : {len(files)}")
    print(f"  Chunks     : {len(chunks)}")
    print(f"  Vectors    : {stored}")
    print(f"  Namespace  : {namespace}")
    print("========================================\n")


if __name__ == "__main__":
    # only run when this file is executed directly
    main()
