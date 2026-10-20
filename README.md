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
