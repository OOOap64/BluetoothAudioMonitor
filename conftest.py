import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

_SLOW_TESTS = {
    'test_stress_500_sequential_reads',
    'test_stress_parallel_reads',
    'test_2000_async_calls_use_one_worker',
    'test_stop_cancels_queued_items',
}
_TRAINING_CLASSES = {'TrainingTests'}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name == 'test_main_gui.py':
            item.add_marker(pytest.mark.gui)
        if (item.path.name == 'test_sound_model.py' and
                item.cls is not None and
                item.cls.__name__ in _TRAINING_CLASSES):
            item.add_marker(pytest.mark.training)
        if item.name in _SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
