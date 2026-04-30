"""
Factory module for language model clients.

This file centralizes the logic for instantiating the correct LLM
based on a simple provider flag. Changing the single `LLM_PROVIDER`
constant in `config.py` will cause the entire application to switch
between Groq and Gemini models without modifying any other code.

The `get_llm` function returns a configured LangChain chat-model
instance and prints diagnostic information about which provider is
being used. It encapsulates error handling so that agents can simply
call `get_llm()` during initialization.
"""

from typing import Any

# import configuration constants used by the factory
from config import (
    LLM_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)

# import both possible LangChain wrappers so they are available
# when the provider selection is made. These imports are lightweight
# and won't be used until the corresponding branch is executed.
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm() -> Any:
    """Return a chat model instance according to the configured provider.

    The function reads :data:`LLM_PROVIDER` from :mod:`config` and
    returns either a :class:`ChatGroq` instance or a
    :class:`ChatGoogleGenerativeAI` instance. If the provider is
    unrecognized, a :class:`ValueError` is raised. Initialization is
    wrapped in a try/except to provide clear logging on failure.

    Returns
    -------
    Any
        An object conforming to the LangChain chat model interface.

    Raises
    ------
    ValueError
        If ``LLM_PROVIDER`` is not one of ``"groq"`` or ``"gemini"``.
    Exception
        Propagates any error raised while constructing the model client.
    """

    provider = (LLM_PROVIDER or "").strip().lower()

    if provider == "groq":
        print("[INFO] LLM factory selected provider 'groq'")
        try:
            return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        except Exception as e:
            print(f"[ERROR] Failed to initialize Groq LLM: {e}")
            raise
    elif provider == "gemini":
        print("[INFO] LLM factory selected provider 'gemini'")
        try:
            return ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
        except Exception as e:
            print(f"[ERROR] Failed to initialize Gemini LLM: {e}")
            raise
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. ``groq`` or ``gemini`` expected."
        )
