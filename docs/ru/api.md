# Python API

## Публичные импорты

Корень пакета экспортирует поддерживаемые классы, не импортируя Drake до первого
обращения к объекту, которому он нужен:

```python
from online_gcs import (
    GCSPathPlanner,
    IRISRegionBuilder,
    OnlineGCS,
    OnlineGCSStats,
    RRTStarPlanner,
    SceneType,
)
```

`OnlineGCS` — высокоуровневый интерфейс. Низкоуровневые планировщики и построитель
областей доступны исследовательскому коду, которому нужен явный контроль, но детали
их методов могут меняться до версии 1.0.

## Онлайн-планировщик

::: online_gcs.OnlineGCS
    options:
      members:
        - run
        - get_statistics
        - save_regions

## Статистика

::: online_gcs.OnlineGCSStats

## Планировщик GCS

::: online_gcs.GCSPathPlanner
    options:
      members:
        - from_iris_builder
        - from_yaml
        - solve_from_configs

## Планировщик RRT*

::: online_gcs.RRTStarPlanner
    options:
      members:
        - plan
        - plan_bidirectional

## Построитель областей IRIS

::: online_gcs.IRISRegionBuilder
    options:
      members:
        - build_regions
        - build_regions_from_seeds
        - save_regions
        - load_regions

## Сцены

::: online_gcs.SceneType
