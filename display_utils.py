"""Rich HTML display utilities for the conditional Poisson sampling notebook."""

from IPython.display import HTML, display
import numpy as np


def html_table(headers, rows, fmt=None, caption=None):
    """Render a list of rows as a styled HTML table.

    Parameters
    ----------
    headers : list of str (may contain HTML/math)
    rows : list of lists
    fmt : list of callables, one per column (optional)
    caption : str (optional)
    """
    css = (
        "style='border-collapse:collapse; font-size:14px; margin:8px 0;'"
    )
    th = "style='border-bottom:2px solid #333; padding:6px 12px; text-align:left;'"
    td = "style='padding:4px 12px; border-bottom:1px solid #ddd;'"
    html = [f"<table {css}>"]
    if caption:
        html.append(f"<caption style='font-weight:bold; margin-bottom:4px;'>{caption}</caption>")
    html.append("<thead><tr>")
    for h in headers:
        html.append(f"<th {th}>{h}</th>")
    html.append("</tr></thead><tbody>")
    for row in rows:
        html.append("<tr>")
        for j, val in enumerate(row):
            if fmt and j < len(fmt) and fmt[j] is not None:
                val = fmt[j](val)
            html.append(f"<td {td}>{val}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    display(HTML("\n".join(html)))


def check_mark(ok):
    return "✓" if ok else "✗"


def poly_html(coeffs, var="z"):
    """Format a polynomial coefficient array as an HTML string."""
    terms = []
    for k, c in enumerate(coeffs):
        if c == 0:
            continue
        if k == 0:
            terms.append(f"{c:g}")
        elif k == 1:
            if c == 1:
                terms.append(var)
            else:
                terms.append(f"{c:g}{var}")
        else:
            if c == 1:
                terms.append(f"{var}<sup>{k}</sup>")
            else:
                terms.append(f"{c:g}{var}<sup>{k}</sup>")
    return " + ".join(terms) if terms else "0"
