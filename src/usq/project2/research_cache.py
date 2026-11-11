"""Completion markers are written last; existence of a data file is insufficient."""
from __future__ import annotations
import json
from pathlib import Path
from .extensions import digest


def atomic_json(path,payload):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(path)


def bundle_complete(root,marker,signature,required):
    root=Path(root);marker=root/marker
    try:
        value=json.loads(marker.read_text())
        if value.get('complete') is not True or value.get('signature')!=signature or set(value.get('files',{}))!=set(required):return False
        return all((root/p).is_file() and digest(root/p)==h for p,h in value['files'].items())
    except (OSError,ValueError,KeyError,TypeError):return False


def finish_bundle(root,marker,signature,required):
    root=Path(root);files={name:digest(root/name) for name in required}
    atomic_json(root/marker,{'signature':signature,'files':files,'complete':True})
