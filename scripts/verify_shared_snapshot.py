"""Verify the published aggregate-result snapshot before editing it."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'PUBLISH_MANIFEST.json').read_text())
failed = []
for item in manifest['files']:
    path = root/item['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        failed.append(item['path'])
if failed:
    raise SystemExit('Changed or missing since initial publication: '+', '.join(failed))
print(f"Verified {len(manifest['files'])} files in the initial Team 34 publication.")
