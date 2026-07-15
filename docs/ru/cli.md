# Справочник CLI

## Точки входа

Установленная команда и точка входа модуля эквивалентны:

```bash
online-gcs --help
python -m online_gcs --help
```

Ошибки аргументов и выполнения дают короткое сообщение и ненулевой код возврата.

## Минимальный запуск

```bash
online-gcs --scene SINGLE_SHELF --iterations 1 --keypoints 2 --prune-interval 0 --seed 42
```

| Параметр | Значение |
| --- | --- |
| `--scene` | `SINGLE_SHELF`, `TWO_SHELVES` или `TABLE_THREE_SHELVES` |
| `--iterations` | Число онлайн-запросов; не меньше 1 |
| `--keypoints` | Число ключевых точек запасного пути; не меньше 2 |
| `--seed` | Random seed поддерживаемых компонентов планировщика |
| `--prune-interval` | Интервал удаления областей; `0` отключает pruning |
| `--warmstart` | Построить начальные области IRIS до цикла запросов |
| `--warmstart-seeds` | Число seed для warm start |
| `--smart-keypoints` | Включить экспериментальный выбор ключевых точек |
| `--k-shortest-paths` | Пути-кандидаты для выбора подграфа GCS |
| `--verbose` | Включить дополнительные сообщения планировщика |

## Режимы планировщика

Одновременно используйте только один baseline-режим:

```bash
online-gcs --scene SINGLE_SHELF --iterations 1 --rrt-only
online-gcs --scene SINGLE_SHELF --iterations 1 --opt-only
```

`--rrt-only` запускает семплирующий baseline. `--opt-only` оптимизирует запасные
пути и поэтому зависит от наличия solver.

## Визуализация и результаты

```bash
mkdir -p artifacts
online-gcs --scene SINGLE_SHELF --iterations 1 --output artifacts/regions.yaml
```

Добавьте `--visualize`, чтобы получить локальный URL Meshcat. Файлы областей и
другие сгенерированные данные следует хранить в игнорируемых каталогах `results/`
или `artifacts/`. Параллельное исследование включается через
`--parallel --num-workers N`; используйте его только там, где уже проверены
multiprocessing и сборка сцены Drake.
