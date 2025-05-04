import os
from functools import lru_cache  # Import caching function
from .ciftools_logger import logger  # Import logger
from pathlib import Path


@lru_cache(maxsize=1)
def find_repo_home() -> str:
    """
    Recursively searches for the repository root by looking for '.git', 'pyproject.toml', or 'setup.py'.
    Uses caching to avoid redundant filesystem lookups.

    Returns:
        str: Absolute path of the repository root.
    """
    current_dir = os.path.abspath(__file__)

    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root "/"
        if any(
            os.path.exists(os.path.join(current_dir, marker))
            for marker in [".git", "pyproject.toml", "setup.py", ".gitignore"]
        ):
            logger.info("Repository root found: %s", current_dir)
            return current_dir  # Found the repo root
        current_dir = os.path.dirname(current_dir)  # Move up one level

    logger.error(
        "Repository root not found. Ensure you're running the script inside a valid project."
    )
    raise RuntimeError(
        "Repository root not found. Make sure you are running the script inside a valid project."
    )
