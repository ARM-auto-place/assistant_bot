# AI-ассистент ARM Auto Place

Python/FastAPI-сервис для Go backend. Отвечает по проверенным FAQ через DeepSeek, при нехватке информации предлагает оператора.

## Файлы

| Файл | Назначение |
|---|---|
| `main.py` | HTTP API, JSON-RPC и резервные ответы |
| `deepseek_client.py` | Запросы к DeepSeek и проверка ответа |
| `db.py` | Подключение к PostgreSQL |
| `faq.py` | Исходные FAQ и точный поиск |
| `faq_repository.py` | Чтение FAQ из базы |
| `seed_faq.py` | Загрузка и обновление FAQ в базе |
| `embeddings.py` | Создание векторов текста |
| `index_faq.py` | Запись векторов FAQ в базу |
| `check_search.py` | Проверка поиска по смыслу |
| `check_deepseek.py` | Проверка ответов DeepSeek |
| `compose.yaml` | Запуск PostgreSQL с pgvector |
| `requirements.txt` | Зависимости Python |
| `.gitignore` | Исключения для Git |
| `README.md` | Краткое описание проекта |

## API

- `GET /health` — проверка работы сервиса.
- `POST /chat` — обычный HTTP-запрос.
- `POST /rpc` — JSON-RPC 2.0, метод `chat`.

Пример запроса к `/rpc`:

```json
{"jsonrpc":"2.0","id":"1","method":"chat","params":{"message":"Сколько стоит продвижение?"}}
```

В `result` возвращаются `answer` и `type_id`:
**1** — DeepSeek, **2** — готовый FAQ, **3** — предложение оператора,
**4** — уточнение (зарезервировано).

Go проверяет поле `error` даже при HTTP 200.

## Текущая версия

DeepSeek получает все 15 FAQ из базы без истории переписки.
Векторный поиск подготовлен, но пока не участвует в выборе ответа API.
При сбое DeepSeek используется точное совпадение с FAQ; иначе предлагается оператор.

Ключ DeepSeek и настройки базы хранятся в локальном `.env`, который не включается в Git.


## Установка и автоматические проверки

Python 3.11 или новее. API и тесты устанавливаются без Torch и загрузки модели векторного поиска:

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy .
pytest -v --cov --cov-report=term-missing
```

GitHub Actions запускает тот же набор проверок на push и pull request в `master` и `develop`.
Используется общий workflow `ARM-auto-place/arm-auto-place/.github/workflows/python-ci.yml@master`.
Для закрытого репозитория общий workflow должен быть доступен этому репозиторию.

Тесты проверяют HTTP `/chat`, JSON-RPC, валидацию ответов DeepSeek, резервные FAQ,
сбои базы/LLM и параметры векторизации. HTTP DeepSeek, PostgreSQL и модель эмбеддингов
заменены локальными подстановками; случайные реальные сетевые вызовы блокируются.
Ключ API, запущенная БД и веса модели для этих тестов не нужны. Это проверки кода
и контракта, а не оценка качества ответов настоящей модели или доступности базы.
Минимальное покрытие основных модулей — 85%.

Для запуска API достаточно `python -m pip install -e .`. Векторный поиск пока не участвует
в обработке `/chat`; для `index_faq.py` и `check_search.py` дополнительно установите:

```bash
python -m pip install -e ".[embeddings]"
```

Существующая команда `python -m pip install -r requirements.txt` по-прежнему устанавливает
полный набор, включая `sentence-transformers`. `requirements-core.txt` — зависимости API,
а `requirements-dev.txt` устанавливает проект с набором `dev`.
