"""Single source of truth for the NeuroWeave Cortex version.

This module must stay dependency-free: setuptools reads it through
``[tool.setuptools.dynamic]`` and ``scripts/check_version.py`` parses it with
``ast`` without importing the package. Never import anything here — importing
``star_graph`` would pull in the whole runtime just to read one string.
"""

__version__ = "1.4.3"

__version_info__ = (1, 4, 3)
