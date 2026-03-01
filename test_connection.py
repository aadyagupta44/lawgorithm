"""
Connection validation script for CodeMind project.

This script performs a series of sanity checks to ensure that
environment variables are available and that the application can
successfully connect to the Groq LLM service and the Pinecone vector
database. It provides clear success/failure messages and handles
errors gracefully so developers can diagnose configuration issues
before running the full application.

Usage:
    python test_connection.py
"""

# import the configuration values from the project config module
from config import GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# ChatGroq is the LangChain wrapper around the Groq API
from langchain_groq import ChatGroq

# Pinecone client for interacting with the vector database service
from pinecone import Pinecone


def main() -> None:
    """Run all connection tests and print results to stdout."""

    # ----- Test 1: Environment key validation -----
    print("\n=== Test 1: Environment Key Validation ===")

    # check if groq key was loaded from .env file
    if GROQ_API_KEY:
        print("[OK] GROQ_API_KEY is set")
    else:
        print("[ERROR] GROQ_API_KEY is missing — check your .env file")

    # check if pinecone key was loaded from .env file
    if PINECONE_API_KEY:
        print("[OK] PINECONE_API_KEY is set")
    else:
        print("[ERROR] PINECONE_API_KEY is missing — check your .env file")

    # ----- Test 2: Groq LLM connection -----
    print("\n=== Test 2: Groq LLM Connection ===")
    try:
        # instantiate the Groq chat model using the provided key and model name
        chat: ChatGroq = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile"
        )

        # send a simple message to verify the connection works
        response = chat.invoke("Say hello and tell me in one sentence what LangChain is.")

        # print just the text content of the response
        print("[OK] Received response from Groq:")
        print(response.content)

    except Exception as e:
        # if anything goes wrong print the exact error so we can debug
        print(f"[ERROR] Failed to communicate with Groq LLM: {e}")

    # ----- Test 3: Pinecone connection -----
    print("\n=== Test 3: Pinecone Connection ===")
    try:
        # initialize pinecone client using the new Pinecone class style
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # get list of all existing indexes in our pinecone account
        existing_indexes = [index.name for index in pc.list_indexes()]

        # check if our codemind index exists
        if PINECONE_INDEX_NAME in existing_indexes:
            print(f"[OK] Pinecone index '{PINECONE_INDEX_NAME}' found and ready")
        else:
            print(f"[ERROR] Index '{PINECONE_INDEX_NAME}' not found")
            print(f"Available indexes: {existing_indexes}")

    except Exception as e:
        # if anything goes wrong print the exact error so we can debug
        print(f"[ERROR] Failed to connect to Pinecone: {e}")


if __name__ == "__main__":
    # only run main() when this file is executed directly
    # not when it is imported by another file
    main()
