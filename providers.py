import asyncio
import time

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from metrics import estimate_tokens

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
BACKOFFS = (1.0, 2.0, 4.0)

CHAT_INSTRUCTION = (
    "You are a code completion engine. Complete the missing middle part of the code below. "
    "Output ONLY the missing code, exactly as it belongs between the prefix and the suffix. "
    "No explanations, no markdown, no code fences, no repeated prefix or suffix."
)


class ProviderError(Exception):
    def __init__(self, kind, message, attempts=1):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.attempts = attempts


def is_retryable(error):
    status = getattr(error, "status_code", None)
    if status in RETRYABLE_STATUS:
        return True
    return isinstance(error, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError,
                              APITimeoutError, APIConnectionError, RateLimitError))


def split_endpoint(url, path):
    if url.endswith(path):
        base = url[: -len(path)]
        return base.rstrip("/") + "/", path
    return url.rstrip("/") + "/", path


class Provider:
    def __init__(self, model_config):
        self.cfg = model_config
        base_url, _ = split_endpoint(model_config.endpoint, "/chat/completions")
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=model_config.api_key or "none",
            timeout=model_config.timeout,
            max_retries=0,
            default_headers=model_config.headers or None,
        )

    def _extra_body(self, with_usage):
        body = dict(self.cfg.extra_body or {})
        if with_usage and "openrouter.ai" in self.cfg.endpoint:
            body.setdefault("usage", {"include": True})
        return body

    async def _with_retries(self, call):
        last_error = None
        for attempt in range(3):
            try:
                return await call(), attempt + 1
            except Exception as e:
                if not is_retryable(e) or attempt == 2:
                    raise ProviderError(type(e).__name__, str(e), attempts=attempt + 1)
                last_error = e
                await asyncio.sleep(BACKOFFS[attempt])
        raise ProviderError(type(last_error).__name__, str(last_error), attempts=3)

    def _empty_result(self):
        return {"prompt_tokens": None, "completion_tokens": None, "live_cost": None, "ttft_ms": None}

    def _tokens(self, result, usage, fallback_text):
        if usage:
            result["prompt_tokens"] = usage.prompt_tokens
            result["completion_tokens"] = usage.completion_tokens
            result["live_cost"] = getattr(usage, "cost", None)
        if result["prompt_tokens"] is None:
            result["prompt_tokens"] = None
        if result["completion_tokens"] is None:
            result["completion_tokens"] = estimate_tokens(fallback_text) if fallback_text else None
        return result

    async def _stream(self, create_call):
        start = time.perf_counter()
        ttft = None
        parts = []
        usage = None
        final_error = None
        for attempt in range(3):
            ttft = None
            parts = []
            try:
                stream = await create_call()
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if getattr(chunk, "choices", None):
                        delta = getattr(chunk.choices[0], "delta", None)
                        text = getattr(delta, "content", None) if delta else None
                        if not text:
                            text = getattr(chunk.choices[0], "text", None)
                        if text:
                            if ttft is None:
                                ttft = (time.perf_counter() - start) * 1000
                            parts.append(text)
                break
            except Exception as e:
                if not is_retryable(e) or attempt == 2:
                    raise ProviderError(type(e).__name__, str(e), attempts=attempt + 1)
                final_error = e
                await asyncio.sleep(BACKOFFS[attempt])
        result = self._empty_result()
        result["text"] = "".join(parts)
        self._tokens(result, usage, result["text"])
        if not parts:
            result["prompt_tokens"] = None
        result["ttft_ms"] = ttft
        result["attempts"] = attempt + 1
        return result

    async def chat(self, prefix, suffix, stream):
        messages = [
            {"role": "system", "content": CHAT_INSTRUCTION},
            {"role": "user", "content": f"{prefix}{suffix}"},
        ]
        kwargs = dict(
            model=self.cfg.model,
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        extra = self._extra_body(True)
        if extra:
            kwargs["extra_body"] = extra

        def request():
            return self.client.chat.completions.create(**kwargs)

        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            result = await self._stream(request)
            if result["prompt_tokens"] is None:
                result["prompt_tokens"] = estimate_tokens(prefix + suffix)
            return result
        response, attempts = await self._with_retries(request)
        result = self._empty_result()
        result["text"] = response.choices[0].message.content or ""
        self._tokens(result, response.usage, result["text"])
        if result["prompt_tokens"] is None:
            result["prompt_tokens"] = estimate_tokens(prefix + suffix)
        if result["completion_tokens"] is None:
            result["completion_tokens"] = estimate_tokens(result["text"])
        result["ttft_ms"] = None
        result["attempts"] = attempts
        return result

    async def fim(self, prefix, suffix, stream):
        protocol = self.cfg.fim_protocol
        if protocol == "deepseek_fim":
            kwargs = dict(model=self.cfg.model, prompt=prefix, suffix=suffix,
                          max_tokens=min(self.cfg.max_tokens, 4096),
                          temperature=self.cfg.temperature)
            path = "completions"
        elif protocol == "openai_completions":
            prompt = self.cfg.fim_template
            if prompt:
                prompt = prompt.replace("{prefix}", prefix).replace("{suffix}", suffix)
                kwargs = dict(model=self.cfg.model, prompt=prompt,
                              max_tokens=self.cfg.max_tokens, temperature=self.cfg.temperature)
            else:
                kwargs = dict(model=self.cfg.model, prompt=prefix, suffix=suffix,
                              max_tokens=self.cfg.max_tokens, temperature=self.cfg.temperature)
            path = "completions"
        elif protocol == "llamacpp_infill":
            return await self._llamacpp_infill(prefix, suffix)
        else:
            raise ProviderError("config", f"unknown fim_protocol: {protocol}")
        return await self._fim_openai(kwargs, path, stream)

    async def _fim_openai(self, kwargs, path, stream):
        def request():
            if path == "completions":
                return self.client.completions.create(**kwargs)
            raise ProviderError("config", f"unknown fim path: {path}")

        if stream:
            kwargs["stream"] = True
            result = await self._stream(request)
            return result
        response, attempts = await self._with_retries(request)
        result = self._empty_result()
        result["text"] = response.choices[0].text or ""
        self._tokens(result, response.usage, result["text"])
        if result["completion_tokens"] is None:
            result["completion_tokens"] = estimate_tokens(result["text"])
        result["ttft_ms"] = None
        result["attempts"] = attempts
        return result

    async def _llamacpp_infill(self, prefix, suffix):
        payload = {
            "input_prefix": prefix,
            "input_suffix": suffix,
            "n_predict": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "cache_prompt": True,
        }
        headers = dict(self.cfg.headers or {})
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        start = time.perf_counter()
        last_error = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
                    response = await client.post(self.cfg.fim_endpoint, json=payload, headers=headers)
                if response.status_code in RETRYABLE_STATUS:
                    last_error = ProviderError("http", f"HTTP {response.status_code}: {response.text[:300]}", attempts=attempt + 1)
                    await asyncio.sleep(BACKOFFS[attempt])
                    continue
                if response.status_code != 200:
                    raise ProviderError("http", f"HTTP {response.status_code}: {response.text[:300]}", attempts=attempt + 1)
                data = response.json()
                result = self._empty_result()
                result["text"] = data.get("content", "")
                result["ttft_ms"] = (time.perf_counter() - start) * 1000
                result["attempts"] = attempt + 1
                return result
            except ProviderError:
                raise
            except Exception as e:
                last_error = ProviderError(type(e).__name__, str(e), attempts=attempt + 1)
                await asyncio.sleep(BACKOFFS[attempt])
        if last_error is None:
            last_error = ProviderError("http", "request failed")
        raise last_error

    async def close(self):
        await self.client.close()
