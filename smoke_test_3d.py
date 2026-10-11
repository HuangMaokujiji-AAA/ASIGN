"""Run all four folds briefly without editing the full training configuration."""

import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    source_path = Path(__file__).resolve().with_name('main_3d.py')
    source = source_path.read_text(encoding='utf-8')
    Path('weight').mkdir(exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix='smoke_', dir='weight'))
    changes = (
        ("save_path = './weight/ours'", f'save_path = {str(output_dir)!r}', 1),
        ('max_epoch = 51', 'max_epoch = 1', 1),
        ('in tmp_loader:', 'in itertools.islice(tmp_loader, 2):', 3),
    )
    for old, new, expected in changes:
        if source.count(old) != expected:
            raise RuntimeError(f'Training code changed; smoke replacement failed: {old}')
        source = source.replace(old, new)

    sys.argv = [str(source_path), *sys.argv[1:]]
    scope = {'__name__': '__main__', '__file__': str(source_path), 'itertools': itertools}
    exec(compile(source, str(source_path), 'exec'), scope)

    for fold in range(4):
        if not (output_dir / f'model_weights_{fold}.pth').is_file():
            raise RuntimeError(f'Missing checkpoint for fold {fold}')
    if not (output_dir / 'ours_metrics.xlsx').is_file():
        raise RuntimeError('Missing metrics spreadsheet')
    values = scope['df'].to_numpy(dtype=float)
    if values.shape != (8, 3) or not np.isfinite(values).all():
        raise RuntimeError(f'Invalid smoke metrics: shape={values.shape}, values={values}')
    print(f'\nSMOKE TEST PASSED: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
