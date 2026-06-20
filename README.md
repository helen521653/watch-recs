# watch-recs — Персональные рекомендации фильмов и сериалов

Система персональных рекомендаций контента на основе **нейронной коллаборативной фильтрации** ([NCF, He et al., 2017](https://arxiv.org/abs/1708.05031)).

Модель обучается на истории просмотров пользователей и предсказывает, какие фильмы и сериалы им понравятся. Используется **имплицитная обратная связь**: взаимодействие с товаром (рейтинг ≥ 4) трактуется как положительный сигнал, остальное сэмплируется как отрицательный.

---

## Описание задачи

**Данные:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), подмножество Movies & TV.
Датасет собран в период 1996–2023 гг. и содержит оценки товаров Amazon.
Используется 5-core фильтрация (только пользователи и товары с ≥ 5 отзывами) с разбивкой leave-one-out.
Объём: ~657k пользователей, ~198k товаров, ~6.1M взаимодействий, три файла суммарным размером ~2 ГБ.
Каждая запись содержит `user_id`, `parent_asin` (идентификатор товара), `rating` (1–5), `history` (все предыдущие взаимодействия пользователя).

**Задача:** top-K рекомендации — для каждого пользователя предсказать, какой товар ему понравится следующим.

**Метрики:** Precision@10, Recall@10, NDCG@10 — стандартные метрики ранжирования ([torchmetrics.retrieval](https://lightning.ai/docs/torchmetrics/stable/retrieval/normalized_dcg.html)).

**Модели:**
- **NCF** — нейронная коллаборативная фильтрация: эмбеддинги пользователей и товаров конкатенируются и подаются в MLP.
- **Popularity** — базлайн: рекомендует самые популярные товары, которые пользователь ещё не смотрел.

---

## Структура проекта

```
watch-recs/
├── configs/                    # Hydra-конфиги (иерархические)
│   ├── config.yaml             # точка входа (defaults)
│   ├── data/
│   │   └── default.yaml        # параметры датасета и препроцессинга
│   ├── model/
│   │   └── ncf.yaml            # архитектура NCF
│   └── training/
│       └── default.yaml        # параметры обучения, пути, MLflow
├── data/
│   ├── dataset.py              # DataPreprocessor, InteractionDataset, RecsDataModule
│   └── download.py             # загрузка данных (DVC pull / прямое скачивание)
├── models/
│   ├── ncf.py                  # NCFModel (PyTorch Lightning)
│   └── popularity.py           # PopularityRecommender (базовая линия)
├── scripts/
│   ├── train.py                # точка входа для обучения
│   └── export_trt.sh           # конвертация ONNX → TensorRT
├── infer.py                    # точка входа для инференса (публичный API)
├── Movies_and_TV.train.csv.gz.dvc
├── Movies_and_TV.valid.csv.gz.dvc
├── Movies_and_TV.test.csv.gz.dvc
├── pyproject.toml
└── uv.lock
```

---

## Setup

**Требования:** Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone <repo-url>
cd watch-recs
uv sync
uv run pre-commit install
```

---

## Данные

Данные версионируются через [DVC](https://dvc.org/) с локальным remote-хранилищем.

**Вариант 1 — через DVC (если remote доступен):**
```bash
uv run dvc pull --remote data-remote
```

**Вариант 2 — прямое скачивание из открытых источников:**
```bash
uv run python scripts/train.py training.download=true
```

Файлы данных (~2 ГБ суммарно) должны находиться в корне репозитория:
```
Movies_and_TV.train.csv.gz
Movies_and_TV.valid.csv.gz
Movies_and_TV.test.csv.gz
```

---

## Train

**Запуск MLflow-сервера** (в отдельном терминале, перед обучением):
```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

**Обучение NCF с параметрами по умолчанию:**
```bash
uv run python scripts/train.py
```

**Быстрая проверка корректности кода (1 батч, логирование отключено):**
```bash
uv run python scripts/train.py training.fast_dev_run=true
```

**Переопределение гиперпараметров через CLI (Hydra):**
```bash
uv run python scripts/train.py model.embedding_dim=64 model.lr=0.0005 training.max_epochs=20
```

**Обучение с несколькими seeds (Hydra multirun):**
```bash
uv run python scripts/train.py --multirun data.seed=42,123,777
```

После обучения:
- Метрики и графики доступны в **MLflow UI:** `http://127.0.0.1:8080`
- Графики (loss, Precision@10, Recall@10, NDCG@10) сохраняются в папку `plots/`

**Основные параметры конфигурации:**

| Параметр | По умолчанию | Описание |
|---|---|---|
| `model.embedding_dim` | 32 | Размерность эмбеддингов |
| `model.hidden_layers` | [64, 32, 16] | Слои MLP |
| `model.lr` | 0.001 | Learning rate |
| `model.top_k` | 10 | K для метрик ранжирования |
| `data.batch_size` | 1024 | Размер батча |
| `data.neg_sample_ratio` | 4 | Негативных примеров на 1 позитивный (обучение) |
| `data.eval_neg_sample_ratio` | 49 | Негативных кандидатов на валидации/тесте |
| `training.max_epochs` | 10 | Максимальное число эпох |
| `training.early_stopping_patience` | 3 | Patience для early stopping |

---

## Inference

### Production preparation

После обучения скрипт автоматически:
1. Экспортирует модель NCF в формат **ONNX** (`models/ncf.onnx`) через `torch.onnx.export` с динамическим batch size
2. Сохраняет метаданные модели (`models/ncf.json`) — `num_users` и `num_items`
3. Логирует ONNX-модель в MLflow (ран `ncf-onnx`) для последующего Serving

Артефакты поставки:
- `models/ncf.onnx` — модель для инференса
- `models/ncf.json` — метаданные (размеры словарей)

**Конвертация в TensorRT** (опционально, для максимальной скорости на GPU):
```bash
bash scripts/export_trt.sh models/ncf.onnx models/ncf.trt
```
Требует `trtexec` из пакета TensorRT.

### Infer

Входные данные: `user_id` — целочисленный индекс пользователя (0-based, из обучения).

**Запуск инференса через CLI:**
```bash
python infer.py --user_id=42 --top_k=10
```

**Запуск с другим путём к модели:**
```bash
python infer.py --user_id=42 --onnx_path=models/ncf.onnx --top_k=20
```

Вывод — список `item_idx` (целочисленных индексов товаров) отсортированных по убыванию скора.

### MLflow Serving

Запустить REST-сервер для инференса через MLflow:
```bash
# 1. Запустить MLflow сервер (если не запущен)
mlflow server --host 127.0.0.1 --port 8080 \
    --backend-store-uri sqlite:////tmp/mlflow.db \
    --artifacts-destination /tmp/mlflow-artifacts

# 2. Узнать run_id рана ncf-onnx из MLflow UI или CLI
mlflow runs list --experiment-name watch-recs

# 3. Поднять serving
mlflow models serve \
    --model-uri "runs:/<run_id>/model" \
    --port 5002 \
    --no-conda
```

**Пример запроса:**
```bash
curl -X POST http://127.0.0.1:5002/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": {"user_ids": [42, 42], "item_ids": [0, 1]}}'
```
