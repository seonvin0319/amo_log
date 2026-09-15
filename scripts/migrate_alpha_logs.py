#!/usr/bin/env python3
"""Convert stored logs with the same normalizer used by every uploader."""
import argparse
import json
from pathlib import Path
import yaml
import build_catalog
from alpha_logs import SCHEMA, digest, legacy_keys


def audit(root):
    runs = files = unparsed = 0
    for section in ('main', 'ablation'):
        for mp in (root/section/'td3_amo').rglob('run_meta.json'):
            m = json.loads(mp.read_text())
            record = m.get('scale_conversion', {})
            if record.get('schema') != SCHEMA or list(legacy_keys(m['settings'])):
                raise ValueError('Unconverted run: '+str(mp))
            for path in sorted(mp.parent.iterdir()):
                if path.name=='run_meta.json' or path.suffix not in ('.json','.jsonl','.yaml','.yml'):
                    continue
                item = record['files'].get(path.name, {})
                if digest(path) != item.get('sha256'):
                    raise ValueError('Conversion hash mismatch: '+str(path))
                with path.open() as src:
                    if path.suffix=='.jsonl':
                        rows=(json.loads(line) for line in src if line.strip())
                    else:
                        rows=[yaml.safe_load(src) if path.suffix in ('.yaml','.yml') else json.load(src)]
                    for row in rows:
                        if isinstance(row,dict) and 'legacy_unparsed_record' in row: unparsed += 1
                        if list(legacy_keys(row)):
                            raise ValueError('Unconverted T field: '+str(path))
                files += 1
            runs += 1
    print(json.dumps({'schema':SCHEMA,'verified_td3_runs':runs,'verified_files':files,'preserved_unparsed_rows':unparsed}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if not args.check:
        build_catalog.build()
    audit(root)


if __name__=='__main__':main()
