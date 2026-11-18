"""CPU checks for H dev adapter ordering and checkpoint/archive bindings."""
from concurrent.futures import ThreadPoolExecutor
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from go_aicup.tasks.strength.policy_h import dev_eval as h


def test_two_worker_merge_preserves_game_order_and_lengths():
    dev = pd.DataFrame({'game_index': [91, 12, 53, 4], 'length': [3, 5, 2, 4]})
    original_temporary = tempfile.TemporaryDirectory

    def fake_worker(rank, part, config, directory):
        from pathlib import Path
        values = np.repeat(part.game_index.to_numpy(), part.length.to_numpy())
        for name in ('logp', 'q', 'target', 'side'):
            np.save(Path(directory) / f'{rank}_{name}.npy', values)
        return np.r_[0, np.cumsum(part.length)], {'peak_gpu_gib': rank + 1}

    with patch.object(h.torch.cuda, 'device_count', return_value=2), \
         patch.object(h, 'ProcessPoolExecutor', side_effect=lambda *a, **kw: ThreadPoolExecutor(2)), \
         patch.object(h, 'extract_worker', side_effect=fake_worker), \
         patch.object(h.tempfile, 'TemporaryDirectory', side_effect=lambda **kw: original_temporary(dir='/tmp')):
        bounds, arrays, stats = h.extract(dev, {}, None)
    np.testing.assert_array_equal(bounds, [0, 3, 8, 10, 14])
    for value in arrays.values():
        np.testing.assert_array_equal(value, np.repeat(dev.game_index, dev.length))
    assert stats['positions'] == 14 and stats['peak_gpu_gib'] == 2


def test_adapter_uses_h_archive_and_models():
    def check():
        assert h.shared.build is h.h.build
        assert h.shared.BranchHead is h.h.BranchHead
        assert h.shared.finalize_cache is h.h.finalize_cache
        assert h.shared.evaluate_role is h.h.evaluate_role
        assert h.shared.extract is h.extract
        assert h.shared.policy_zip.ARCHIVE == h.h.ARCHIVE
    # Restore module overrides after this test to avoid changing other tests' C bindings.
    names = ('build', 'BranchHead', 'finalize_cache', 'evaluate_role', 'extract', 'teacher_identity')
    with patch.object(h.shared, 'main', side_effect=check), \
         patch.multiple(h.shared, **{name: getattr(h.shared, name) for name in names}), \
         patch.multiple(h.shared.policy_zip, **{name: getattr(h.shared.policy_zip, name) for name in
                        ('NAME', 'ARCHIVE', 'IDENTITY', 'FIELDS', 'selection', 'load')}), \
         patch.object(h.sys, 'argv', ['dev_eval', '--config', 'unused']):
        h.main()
