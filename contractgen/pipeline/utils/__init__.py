"""Utility namespace for the contract-guided pipeline.

Import utility modules directly, for example
`from contractgen.pipeline.utils.ref_repo_clone import clone_reference_repository`.
Keeping this package initializer lightweight prevents optional LLM wrappers from
being imported during CLI argument parsing.
"""

__all__: list[str] = []
