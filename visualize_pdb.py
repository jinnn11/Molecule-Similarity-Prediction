#!/usr/bin/env python3
"""
Generate a standalone HTML file to visualize a PDB using 3Dmol.js (CDN).
Usage:
  python visualize_pdb.py path/to/file.pdb [output.html]
"""

from __future__ import annotations

import html
import pathlib
import sys


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>3D Molecule Viewer</title>
    <style>
      html, body {{ height: 100%; margin: 0; }}
      #viewer {{ width: 100%; height: 100%; }}
    </style>
    <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
  </head>
  <body>
    <div id="viewer"></div>
    <script>
      const pdbText = `{pdb_text}`;
      const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
      viewer.addModel(pdbText, "pdb");
      viewer.setStyle({{}}, {{ stick: {{}}, sphere: {{ scale: 0.25 }} }});
      viewer.zoomTo();
      viewer.render();
    </script>
  </body>
</html>
"""


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python visualize_pdb.py path/to/file.pdb [output.html]")
        return 2

    pdb_path = pathlib.Path(argv[1])
    if not pdb_path.exists():
        print(f"Error: file not found: {pdb_path}")
        return 1

    output_path = pathlib.Path(argv[2]) if len(argv) == 3 else pdb_path.with_suffix(".html")

    pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace")
    # Escape to keep the embedded JS string safe.
    pdb_text = html.escape(pdb_text).replace("\n", "\\n")

    html_out = HTML_TEMPLATE.format(pdb_text=pdb_text)
    output_path.write_text(html_out, encoding="utf-8")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
