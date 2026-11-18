#!/usr/bin/env python3
"""Обучение собственного классификатора звуков.

Пример:

    python tools/train_sound_model.py \
        --dataset data/sound_dataset \
        --output models/sound_model.npz \
        --epochs 300

Структура данных::

    data/sound_dataset/
        speech/*.wav
        yelling/*.wav
        bird_singing/*.wav
        door_opening/*.wav
        fire_burning/*.wav
        noise/*.wav

После обучения модель автоматически подхватывается приложением из
``models/sound_model.npz`` (или из пути в BLUETOOTH_MONITOR_SOUND_MODEL).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.custom_sound_model import scan_dataset, train_model  # noqa: E402


def _progress(fraction, message):
    percent = int(round(max(0.0, min(1.0, fraction)) * 100))
    print(f'[{percent:3d}%] {message}', flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Обучение классификатора звуков (речь, крик, птицы, '
                    'дверь, огонь и т.д.)')
    parser.add_argument(
        '--dataset', default='data/sound_dataset',
        help='каталог с подкаталогами-классами (по умолчанию data/sound_dataset)')
    parser.add_argument(
        '--output', default='models/sound_model.npz',
        help='куда сохранить модель (по умолчанию models/sound_model.npz)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.02)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-per-class', type=int, default=None)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--list-classes', action='store_true')
    parser.add_argument('--report', default=None,
                        help='сохранить отчёт в JSON')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = Path(args.dataset).expanduser()
    if not dataset.is_absolute():
        dataset = PROJECT_ROOT / dataset

    try:
        classes, warnings = scan_dataset(dataset)
    except Exception as exc:
        print(f'Ошибка датасета: {exc}', file=sys.stderr)
        return 2

    print(f'Датасет: {dataset}')
    total = 0
    for name in sorted(classes):
        count = len(classes[name])
        total += count
        print(f'  {name}: {count} файл(ов)')
    print(f'Всего файлов: {total}')
    for warning in warnings:
        print(f'  ! {warning}')
    if args.list_classes:
        return 0

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    try:
        report = train_model(
            dataset,
            output,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            test_size=args.val_split,
            augment=not args.no_augment,
            seed=args.seed,
            progress=_progress,
            max_files_per_class=args.max_per_class,
        )
    except Exception as exc:
        print(f'Ошибка обучения: {exc}', file=sys.stderr)
        return 1

    print()
    print(f"Модель: {report['model_path']}")
    print(f"Точность на проверке: {report['val_accuracy'] * 100:.2f}%")
    print(f"Точность на обучении: {report['train_accuracy'] * 100:.2f}%")
    print('Метрики по классам:')
    for name, metrics in report['per_class'].items():
        print(
            f"  {name}: precision={metrics['precision']:.2f} "
            f"recall={metrics['recall']:.2f} f1={metrics['f1']:.2f} "
            f"(примеров: {metrics['support']})")
    if report['warnings']:
        print('Предупреждения:')
        for warning in report['warnings']:
            print(f'  ! {warning}')

    if args.report:
        report_path = Path(args.report).expanduser()
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        print(f'Отчёт: {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
