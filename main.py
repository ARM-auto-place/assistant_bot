import json
from enum import IntEnum

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

import logging
import os
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from psycopg import Error as DatabaseError
from pydantic import BaseModel, ConfigDict, Field

from deepseek_client import ask_deepseek
from faq import FAQS, normalize_question
from faq_repository import get_all_faqs


load_dotenv(Path(__file__).with_name(".env"))

logger = logging.getLogger(__name__)

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

app = FastAPI(
    title=os.getenv("APP_NAME", "auto-assistant"),
    version="0.2.0",
    default_response_class=UTF8JSONResponse,
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )


class ChatResponse(BaseModel):
    status: Literal["answer", "clarification", "handoff"]
    answer: str
    offer_operator: bool
    reason: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None


def handoff(request: ChatRequest, reason: str) -> ChatResponse:
    return ChatResponse(
        status="handoff",
        answer=(
            "Не могу дать проверенный ответ на этот вопрос. "
            "Хотите обратиться к оператору?"
        ),
        offer_operator=True,
        reason=reason,
        session_id=request.session_id,
    )


def fallback(
    request: ChatRequest,
    faqs: list[dict],
    reason: str,
) -> ChatResponse:
    question = normalize_question(request.message)

    for faq in faqs:
        if normalize_question(faq["question"]) == question:
            return ChatResponse(
                status="answer",
                answer=faq["answer"],
                offer_operator=False,
                reason=reason,
                source_ids=[faq["id"]],
                session_id=request.session_id,
            )

    return handoff(request, reason)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        faqs = get_all_faqs()
    except DatabaseError as error:
        logger.warning(
            "Database unavailable: %s",
            type(error).__name__,
        )
        return fallback(
            request,
            FAQS,
            reason="database_unavailable",
        )

    if not faqs:
        return handoff(request, reason="knowledge_base_empty")

    try:
        result = ask_deepseek(request.message, faqs)

    except httpx.HTTPStatusError as error:
        logger.warning(
            "DeepSeek HTTP status: %s",
            error.response.status_code,
        )
        return fallback(
            request,
            faqs,
            reason="llm_unavailable",
        )

    except httpx.RequestError as error:
        logger.warning(
            "DeepSeek connection error: %s",
            type(error).__name__,
        )
        return fallback(
            request,
            faqs,
            reason="llm_unavailable",
        )

    except (ValueError, KeyError, IndexError, TypeError) as error:
        logger.warning(
            "DeepSeek configuration or response error: %s",
            type(error).__name__,
        )
        return fallback(
            request,
            faqs,
            reason="llm_invalid_response_or_config",
        )

    return ChatResponse(
        status=result.status,
        answer=result.answer,
        offer_operator=result.status == "handoff",
        reason=(
            "insufficient_information_or_operator_requested"
            if result.status == "handoff"
            else None
        ),
        source_ids=result.source_ids,
        session_id=request.session_id,
    )

class AnswerType(IntEnum):
    AI = 1
    FAQ = 2
    HANDOFF = 3
    CLARIFICATION = 4


def get_answer_type(result: ChatResponse) -> AnswerType:
    if result.status == "handoff":
        return AnswerType.HANDOFF

    if result.status == "clarification":
        return AnswerType.CLARIFICATION

    # В текущем /chat резервные ответы всегда имеют reason.
    if result.reason is not None:
        return AnswerType.FAQ

    return AnswerType.AI


def rpc_error(request_id, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


class RPCChatParams(ChatRequest):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        strict=True,
    )


async def process_rpc_call(payload):
    if not isinstance(payload, dict):
        return rpc_error(None, -32600, "Invalid Request")

    request_id = payload.get("id")

    valid_id = (
        request_id is None
        or type(request_id) in (str, int, float)
    )

    if (
        payload.get("jsonrpc") != "2.0"
        or not isinstance(payload.get("method"), str)
        or not valid_id
        or (
            "params" in payload
            and not isinstance(payload["params"], (dict, list))
        )
    ):
        return rpc_error(None, -32600, "Invalid Request")

    # Без id это уведомление: ответ на него не отправляем.
    notification = "id" not in payload

    if payload["method"] != "chat":
        result = rpc_error(request_id, -32601, "Method not found")
    else:
        try:
            # Для метода chat используем именованные параметры.
            params = RPCChatParams.model_validate(
                payload.get("params", {})
            )
        except ValidationError:
            result = rpc_error(request_id, -32602, "Invalid params")
        else:
            try:
                # chat выполняет синхронные запросы к БД и DeepSeek.
                # Выносим его в поток, чтобы не блокировать сервер.
                answer = await run_in_threadpool(chat, params)

                result = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "answer": answer.answer,
                        "type_id": int(get_answer_type(answer)),
                    },
                }
            except Exception as error:
                logger.error(
                    "RPC chat failed: %s",
                    type(error).__name__,
                )
                result = rpc_error(
                    request_id,
                    -32603,
                    "Internal error",
                )

    return None if notification else result


def reject_json_constant(value):
    raise ValueError("Invalid JSON constant")


@app.post("/rpc")
async def rpc_endpoint(request: Request):
    try:
        payload = json.loads(
            await request.body(),
            parse_constant=reject_json_constant,
        )
    except (ValueError, UnicodeDecodeError):
        return UTF8JSONResponse(
            rpc_error(None, -32700, "Parse error")
        )

    if isinstance(payload, list):
        if not payload:
            return UTF8JSONResponse(
                rpc_error(None, -32600, "Invalid Request")
            )

        responses = []

        for item in payload:
            result = await process_rpc_call(item)
            if result is not None:
                responses.append(result)

        if not responses:
            return Response(status_code=204)

        return UTF8JSONResponse(responses)

    result = await process_rpc_call(payload)

    if result is None:
        return Response(status_code=204)

    return UTF8JSONResponse(result)