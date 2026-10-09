"""Let pytest on the host import the workspace packages straight from source."""

from pathlib import Path
import sys

for package in ('did_agent', 'did_judge', 'did_llm'):
    sys.path.insert(0, str(Path(__file__).parent / package))
