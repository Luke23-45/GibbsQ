"""
LaTeX table generation for the GibbsQ analysis framework.

Produces publication-ready LaTeX tables using ``booktabs`` style,
with support for:
- Automatic formatting of scientific notation values.
- PASS/FAIL status highlighting.
- Direct ``.tex`` file export.

All generated tables are standalone-compilable via:

.. code-block:: latex

    \\documentclass{article}
    \\usepackage{booktabs}
    \\begin{document}
    \\input{table.tex}
    \\end{document}

Design notes
------------
* We use standard ``tabular`` wrapped in ``\\resizebox`` when needed.
* System names are escaped for LaTeX safety.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


def format_scientific(value: float, *, precision: int = 2) -> str:
    """Format a value in scientific notation for LaTeX.

    Parameters
    ----------
    value : float
        Value to format.
    precision : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted string, e.g. ``"$1.23 \\times 10^{-4}$"``.
    """
    if np.isnan(value):
        return "—"
    return f"{value:.{precision}e}"


def format_value(
    value: float,
    *,
    precision: int = 4,
    use_math: bool = False,
) -> str:
    """Format a single numeric value for LaTeX.

    Parameters
    ----------
    value : float
        Value to format.
    precision : int
        Number of decimal places.
    use_math : bool
        If True, wrap in ``$...$``.

    Returns
    -------
    str
    """
    if np.isnan(value):
        return "—"
    fmt = f"{{:.{precision}f}}"
    s = fmt.format(value)
    return f"${s}$" if use_math else s


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text.

    Does NOT escape text already containing LaTeX commands
    (detected by presence of backslash).
    """
    if "\\" in text or "$" in text:
        return text  # assume already LaTeX-safe
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


def _status_color(status: str) -> str:
    """Return LaTeX color command for PASS/FAIL status."""
    if status == "PASS":
        return r"\textcolor{green!60!black}{PASS}"
    elif status == "FAIL":
        return r"\textcolor{red!80!black}{FAIL}"
    return status


def save_latex_table(
    tex_str: str,
    path: Union[str, Path],
    *,
    standalone_header: bool = False,
) -> Path:
    """Save a LaTeX table string to a ``.tex`` file.

    Parameters
    ----------
    tex_str : str
        LaTeX source.
    path : str or Path
        Output file path (should end in ``.tex``).
    standalone_header : bool
        If True, prepend ``\\documentclass`` / ``\\usepackage``
        so the file is directly compilable with ``pdflatex``.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = tex_str
    if standalone_header:
        preamble = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{xcolor}\n"
            "\\begin{document}\n\n"
        )
        postamble = "\n\n\\end{document}\n"
        content = preamble + content + postamble

    path.write_text(content, encoding="utf-8")
    logger.info("Saved LaTeX table: %s", path)
    return path.resolve()
