"""Utility to load example inputs from example_inputs.txt

The format expected is blocks separated by a line containing only
"---". Each block may start with comment metadata lines:

# name: Example name
# expected_critical: true|false

Followed by the text body for that example.

Provides: load_example_inputs(path: str = "example_inputs.txt") -> list[dict]
Each dict contains: 'name', 'expected_critical', 'text'
"""

from pathlib import Path
import re
from typing import List


def _parse_block(block: str) -> dict:
    lines = block.strip().splitlines()
    name = None
    expected_critical = False
    body_lines: List[str] = []

    meta_re = re.compile(r"^#\s*(?P<key>[^:]+):\s*(?P<val>.+)")

    for line in lines:
        m = meta_re.match(line)
        if m:
            key = m.group('key').strip().lower()
            val = m.group('val').strip()
            if key == 'name':
                name = val
            elif key == 'expected_critical':
                expected_critical = val.lower() in ('1', 'true', 'yes')
            continue

        # non-meta line -> part of body
        body_lines.append(line)

    text = "\n".join(body_lines).strip()
    if not name:
        # fall back to first 40 chars of text
        name = (text[:40] + '...') if len(text) > 40 else text

    return {
        'name': name,
        'expected_critical': expected_critical,
        'text': text
    }


def load_example_inputs(path: str = "example_inputs.txt") -> list:
    """Load and parse the example inputs file next to this module.

    Returns a list of dicts with keys: name, expected_critical, text
    """
    base = Path(__file__).parent
    txt_path = base / path

    if not txt_path.exists():
        raise FileNotFoundError(f"Example inputs file not found: {txt_path}")

    content = txt_path.read_text(encoding='utf-8')

    # Split blocks on lines that contain only three dashes
    raw_blocks = [b for b in re.split(r"^---$", content, flags=re.MULTILINE) if b.strip()]

    examples = []
    for block in raw_blocks:
        parsed = _parse_block(block)
        # Only include blocks that have non-empty text
        if parsed['text']:
            examples.append(parsed)

    return examples


if __name__ == '__main__':
    examples = load_example_inputs()
    for i, ex in enumerate(examples, 1):
        print(f"Test {i}: {ex['name']} (expected_critical={ex['expected_critical']})")
        print(ex['text'][:200])
        print('-' * 60)
