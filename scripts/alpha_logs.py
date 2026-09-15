#!/usr/bin/env python3
"""Archive TD3-AMO scales in alpha units (alpha = 2 * legacy T).

Only physical scale fields are multiplied. Optimizer settings and diagnostics
are not scales. Historical gradients retain their original coordinates under
legacy_T/, since some old writers logged rho gradients under T names.
"""
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

SCHEMA = 'alpha_v1'
TOKEN = re.compile(r'(?<![A-Za-z0-9])T(?![A-Za-z0-9])')
SCALE = re.compile(r'^T(?:_[EB])?(?:_(?:init|initial|initialization|min|max|raw|effective|eff|used|next|current|before|after|old|new|value|schedule_start|schedule_end|schedule_value))?$')
UNCHANGED = {
    'T_lr', 'T_freq', 'T_schedule', 'T_schedule_steps',
    'T_B_from_T_E_divisor', 'T_B_over_T_E',
    'constrain_T_B_le_T_E', 'project_T_B_to_T_E',
    'T_B_projection_active', 'T_B_projection_count', 'T_projected',
    'delta_log_T_B', 'delta_log_T_E', 'grad_T_B_L1_L2_same_sign',
    'L_T_E', 'L_T_B', 'inner_loss_T', 'inner_loss_T_B',
    'q_abs_mean_T', 'q_abs_mean_T_B', 'critic_T_loss',
}
# These fields identify the historical source; their text is never rewritten.
PROVENANCE = {'git', 'scale_conversion', 'legacy_settings', 'source_path',
              'run_id', 'legacy_name', 'original_rel_path', 'rel_path',
              'alias_of', 'aliases', 'variant', 'family'}


def rule(key, *, config=False):
    if key.startswith('legacy_T/') or not TOKEN.search(key):
        return key, 1
    leaf = key.rsplit('/', 1)[-1]
    if SCALE.fullmatch(leaf):
        return TOKEN.sub('alpha', key), 2
    if leaf in UNCHANGED:
        return TOKEN.sub('alpha', key), 1
    if config or leaf.startswith('T_') and 'grad' not in leaf:
        raise ValueError('Unreviewed legacy T setting: '+key)
    # Loss/gradient diagnostics that do not represent T itself remain raw.
    return 'legacy_T/'+key, 1


def twice(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [twice(v) for v in value]
    if isinstance(value, bool):
        raise ValueError('Boolean recorded as a scale')
    if isinstance(value, (int, float)):
        return value * 2
    if isinstance(value, str):
        return str(float(value) * 2)
    raise ValueError('Non-numeric scale: '+repr(value))


def same(a, b):
    return a == b or isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)


def convert(obj, *, config=False):
    """Idempotent, recursive key/unit conversion; reject conflicting aliases."""
    if isinstance(obj, list):
        return [convert(v, config=config) for v in obj]
    if not isinstance(obj, dict):
        return obj
    result = {}
    for key, value in obj.items():
        if key in PROVENANCE or key.startswith('legacy_T/'):
            new, value = key, value
        else:
            new, factor = rule(key, config=config)
            value = twice(value) if factor == 2 else convert(value, config=config)
        if new in result and not same(result[new], value):
            raise ValueError('Conflicting T/alpha fields: '+key+' -> '+new)
        result[new] = value
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rewrite(path):
    """Stream JSONL and replace each file atomically, preserving untouched lines."""
    import yaml
    changed = False
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.alpha-', delete=False) as dst:
        temp = Path(dst.name)
        try:
            if path.suffix == '.jsonl':
                with path.open() as src:
                    for lineno, line in enumerate(src, 1):
                        if not line.strip():
                            dst.write(line)
                            continue
                        try:
                            original = json.loads(line)
                            updated = convert(original)
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f'{path}:{lineno}: {exc}') from exc
                        different = not same(original, updated)
                        changed |= different
                        dst.write(json.dumps(updated, ensure_ascii=False)+'\n' if different else line)
            else:
                original_text = path.read_text()
                is_yaml = path.suffix in ('.yaml', '.yml')
                original = yaml.safe_load(original_text) if is_yaml else json.loads(original_text)
                updated = convert(original, config=is_yaml)
                changed = not same(original, updated)
                dst.write((yaml.safe_dump(updated, sort_keys=False) if is_yaml else
                           json.dumps(updated, indent=2, sort_keys=True)+'\n') if changed else original_text)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    if changed:
        temp.chmod(path.stat().st_mode)
        temp.replace(path)
    else:
        temp.unlink()
    return changed


def normalize_run(directory, meta):
    """Use per-file hashes, not a run-wide flag: ingestion can overwrite files."""
    if meta.get('method') != 'td3_amo' and meta.get('algo') not in ('amo', 'apart', 'td3_amo'):
        return meta
    meta = dict(meta)
    record = dict(meta.get('scale_conversion', {}))
    if 'T_lr' in meta.get('settings', {}):
        record['meta_lr_source'] = 'T_lr'
    files = dict(record.get('files', {}))
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == 'run_meta.json' or path.suffix not in ('.json', '.jsonl', '.yaml', '.yml'):
            continue
        before = digest(path)
        old = files.get(path.name, {})
        if old.get('sha256') == before:
            continue
        changed = rewrite(path)
        item = {'sha256': digest(path) if changed else before}
        if changed:
            item['legacy_sha256'] = before
        elif 'legacy_sha256' in old:
            item['legacy_sha256'] = old['legacy_sha256']
        files[path.name] = item
    meta['settings'] = convert(meta.get('settings', {}), config=True)
    record.update(schema=SCHEMA, equation='alpha = 2 * T', files=files,
                  diagnostics='legacy_T/ fields retain their original values and coordinates')
    meta['scale_conversion'] = record
    return meta


def legacy_keys(obj):
    """Find unconverted T keys, excluding explicitly preserved diagnostics."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in PROVENANCE or key.startswith('legacy_T/'):
                continue
            if TOKEN.search(key):
                yield key
            yield from legacy_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from legacy_keys(value)
