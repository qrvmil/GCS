# Быстрый старт

## Первый запрос

После [установки](installation.md) запустите один детерминированный запрос в сцене
с одной полкой:

```bash
online-gcs --scene SINGLE_SHELF --iterations 1 --keypoints 2 --prune-interval 0 --seed 42
```

Команда собирает сцену, запускает baseline RRT* для каждого запроса и baseline
TrajOpt при наличии пути, а затем проверяет текущий граф GCS. Если покрытия
недостаточно, fallback-путь даёт ключевые точки для расширения IRIS и повторного GCS.
Сейчас ветка расширения обновляет двунаправленный путь RRT* перед выбором этих точек.

Поэтому стандартный запуск включает вычисления baseline RRT* и обычно TrajOpt даже
при успехе GCS. Для TrajOpt нужен совместимый solver, а IRIS может занимать основную
часть времени запросов с расширением; сравнивайте время только в контролируемых
условиях.

## Python API

```python
from online_gcs import OnlineGCS, SceneType

planner = OnlineGCS(
    scene_type=SceneType.SINGLE_SHELF,
    random_seed=42,
    max_iterations=1,
)
summary = planner.run(num_keypoints=2, prune_interval=0)
print(summary["total_queries"])
```

Возвращаемая сводка содержит счётчики и времена этого запуска. Она не сохраняется,
если пользовательский скрипт явно не записывает её в игнорируемый каталог.

## Примеры

В репозитории есть короткие примеры без визуализации и с Meshcat, использующие
одинаковый seed:

```bash
python examples/minimal_online.py
python examples/visualize_single_shelf.py
```

Пример с визуализацией печатает URL Meshcat и работает до нажатия Ctrl+C.

## Поддерживаемые сцены

| `SINGLE_SHELF` | `TWO_SHELVES` | `TABLE_THREE_SHELVES` |
|:--:|:--:|:--:|
| ![Сцена с одной полкой](../assets/scenes/single-shelf.png) | ![Сцена с двумя полками](../assets/scenes/two-shelves.png) | ![Стол и три полки](../assets/scenes/three-shelves.png) |

Чтобы выбрать другую встроенную сцену, измените только `--scene` (или `SceneType`).
Время работы не обязано совпадать: геометрия столкновений у сцен различается.
