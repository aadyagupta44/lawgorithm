"""
GitHub repository file fetching and loading.

This module provides a `GitHubLoader` class which is responsible for
connecting to GitHub (via PyGithub), enumerating repository files,
filtering by allowed extensions, and extracting readable documentation
content from those files. For Python files it extracts docstrings and
comments, while for text-based files it returns the full file contents.

The class is intentionally conservative: it avoids returning raw
implementation code and focuses on human-readable documentation content
that will be used by the ingestion and embedding pipeline.
"""

from typing import List, Dict, Optional
import re  # regular expressions used to extract comments
import ast  # used to parse Python AST and extract docstrings
from urllib.parse import urlparse  # parse GitHub URL into components
import os  # used for small path handling

from github import Github  # PyGithub client used to fetch repository files

from config import GITHUB_TOKEN, ALLOWED_EXTENSIONS  # import config values


class GitHubLoader:
	"""Class to load documentation content from a GitHub repository.

	This class connects to GitHub using the `GITHUB_TOKEN` from `config`,
	enumerates files in a repository, filters by extension, and returns
	structured dictionaries containing filename, path, content, and metadata.
	"""

	def __init__(self, repo_url: str) -> None:
		"""Initialize the loader and connect to the repository.

		Parameters:
			repo_url: Full GitHub URL (for example https://github.com/user/repo)

		The initializer parses the URL to obtain the owner and repo name,
		initializes the PyGithub client with the token from `config`,
		and stores the repository object for later file access.
		"""

		# store the original URL for debugging and namespace generation
		self.repo_url: str = repo_url  # save the provided GitHub URL

		# parse the URL to extract owner and repository name
		parsed = urlparse(repo_url)  # parse the URL into components
		# path looks like '/owner/repo' so strip leading slash and split
		path_parts = parsed.path.strip("/").split("/")  # split owner and repo

		# conservative default values if parsing fails
		owner = path_parts[0] if len(path_parts) >= 1 else ""
		repo_name = path_parts[1] if len(path_parts) >= 2 else ""

		# initialize PyGithub with token if available (None works in public access)
		try:
			self.gh = Github(GITHUB_TOKEN)  # type: ignore[arg-type]
		except Exception as e:
			# store None and log the error; downstream methods will handle this
			self.gh = None  # type: ignore[assignment]
			print(f"[ERROR] Failed initializing PyGithub client: {e}")

		# attempt to fetch the repository object from GitHub
		self.repo = None  # type: ignore[assignment]
		try:
			if self.gh is not None and owner and repo_name:
				self.repo = self.gh.get_repo(f"{owner}/{repo_name}")
			else:
				self.repo = None
		except Exception as e:
			# do not raise; print clear guidance for debugging
			self.repo = None
			print(f"[ERROR] Could not access repository '{owner}/{repo_name}': {e}")

		# print which repository we attempted to load for user feedback
		print(f"[INFO] Initialized GitHubLoader for repository: {owner}/{repo_name}")

	def get_all_files(self) -> List[Dict[str, str]]:
		"""Fetch and return documentation files from the repository.

		This method recursively walks the repository tree and retrieves
		files that match `ALLOWED_EXTENSIONS`. For Python files it
		extracts docstrings and comments only; for text files it returns
		the full content. The method returns a list of dictionaries with
		keys: filename, filepath, content, file_type, repo_name.

		Returns:
			A list of file metadata dictionaries ready for chunking.
		"""

		results: List[Dict[str, str]] = []  # accumulator for returned file dicts

		# guard: if repo was not initialized, return empty list with message
		if not self.repo:
			print("[WARN] Repository not available; returning empty file list")
			return results

		# helper inner function to recursively walk and collect files
		def _walk_contents(path: str = "") -> None:
			"""Recursively traverse repository contents starting at `path`.

			This inner function mutates the `results` list in the outer scope.
			"""

			try:
				contents = self.repo.get_contents(path)  # may raise on errors
			except Exception as exc:
				# handle read errors gracefully and continue
				print(f"[ERROR] Failed to list contents at path '{path}': {exc}")
				return

			for item in contents:
				# directories should be recursed into
				if item.type == "dir":
					_walk_contents(item.path)  # recurse into subdirectory
					continue

				# only process files
				if item.type != "file":
					continue

				filename = item.name  # filename with extension
				file_ext = os.path.splitext(filename)[1].lower()  # extension like '.py'

				# skip files with extensions not in allowed list
				if file_ext not in ALLOWED_EXTENSIONS:
					# print a small progress message for skipped files
					print(f"[SKIP] Skipping file with unsupported extension: {item.path}")
					continue

				# attempt to read file contents; handle errors per file
				try:
					raw_bytes = item.decoded_content  # bytes of file content
					decoded = raw_bytes.decode("utf-8", errors="replace")  # decode safely
				except Exception as read_err:
					print(f"[ERROR] Could not read file '{item.path}': {read_err}")
					continue

				# extract relevant content depending on file type
				try:
					if file_ext == ".py":
						# extract only docstrings and comments from Python code
						extracted = self.extract_python_docs(decoded)
					else:
						# for markdown and text files, use full content
						extracted = decoded
				except Exception as extract_err:
					# continue on extraction errors but notify the user
					print(f"[ERROR] Failed extracting content from '{item.path}': {extract_err}")
					continue

				# skip empty or trivial content
				if not extracted or extracted.strip() == "":
					print(f"[SKIP] No useful content extracted from '{item.path}'")
					continue

				# assemble the metadata dictionary for this file
				file_dict: Dict[str, str] = {
					"filename": filename,  # e.g., 'README.md'
					"filepath": item.path,  # repository path e.g., 'docs/README.md'
					"content": extracted,  # the text content to ingest
					"file_type": file_ext,  # extension string
					"repo_name": self.repo.name if hasattr(self.repo, "name") else "",
				}

				# append to results and print progress
				results.append(file_dict)
				print(f"[OK] Collected {filename} ({file_ext})")

		# start recursive traversal from repository root
		_walk_contents("")

		print(f"[INFO] Completed fetching files. Total collected: {len(results)}")
		return results

	def extract_python_docs(self, code: str) -> str:
		"""Extract docstrings and comments from Python source code.

		This function uses the `ast` module to parse the code and extract
		module-, class-, and function-level docstrings. It also uses a
		regular expression to capture inline comments that begin with '#'.

		Parameters:
			code: Raw Python source as a string.

		Returns:
			A cleaned string containing the concatenated docstrings and comments.
		"""

		# accumulator for extracted text parts
		parts: List[str] = []

		# extract docstrings via AST safely
		try:
			parsed = ast.parse(code)  # may raise SyntaxError on invalid code
		except Exception as parse_err:
			# if parsing fails, fall back to comment extraction only
			print(f"[WARN] AST parse failed, falling back to comment extraction: {parse_err}")
			# regex fallback below will still attempt to capture comments
			parsed = None  # type: ignore[assignment]

		# module docstring
		try:
			if parsed is not None:
				module_doc = ast.get_docstring(parsed)
				if module_doc:
					parts.append(module_doc)
		except Exception as e:
			print(f"[WARN] Failed to extract module docstring: {e}")

		# walk AST to collect function and class docstrings
		if parsed is not None:
			for node in ast.walk(parsed):
				try:
					if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
						doc = ast.get_docstring(node)
						if doc:
							parts.append(doc)
				except Exception:
					# ignore individual node extraction errors
					continue

		# extract inline comments using regex: lines starting with optional whitespace followed by '#'
		try:
			comment_lines = re.findall(r"(?m)^[ \t]*#.*$", code)
			if comment_lines:
				# clean comment markers and add to parts
				cleaned = [re.sub(r"^[ \t]*#\s?", "", c).strip() for c in comment_lines]
				parts.extend([c for c in cleaned if c])
		except Exception as cre:
			print(f"[WARN] Comment extraction failed: {cre}")

		# join all extracted parts with two newlines for readability when chunking
		final_text = "\n\n".join(parts).strip()
		return final_text

