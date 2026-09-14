import asyncio
import time

import httpx
from openai import AsyncOpenAI

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
BACKOFFS = (1.0, 2.0, 4.0)

CHAT_INSTRUCTION = (
    "You are a code completion engine. Complete the missing middle part of the code below. "
    "Output ONLY the missing code, exactly as it belongs between the prefix and the suffix. "
    "No explanations, no markdown, no code fences, no repeated prefix or suffix."
)


class ProviderError(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
        self.message = message


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
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body
        if "openrouter.ai" in self.cfg.endpoint:
            usage_body = {"usage": {"include": True}}
            usage_body.update(self.cfg.extra_body or {})
            kwargs["extra_body"] = usage_body
        if stream:
            return await self._stream_chat(kwargs)
        response = await self._with_retries(lambda: self.client.chat.completions.create(**kwargs))
        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content or "",
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "live_cost": None,
            "ttft_ms": None,
        }

    async def _stream_chat(self, kwargs):
        start = time.perf_counter()
        ttft = None
        parts = []
        try:
            stream = await self.client.chat.completions.create(**kwargs, stream=True)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    parts.append(chunk.choices[0].delta.content)
        except Exception as e:
            raise ProviderError(type(e).__name__, str(e))
        return {
            "text": "".join(parts),
            "prompt_tokens": None,
            "completion_tokens": None,
            "live_cost": None,
            "ttft_ms": ttft,
        }

    async def fim(self, prefix, suffix, stream):
        if self.cfg.fim_protocol == "deepseek_fim":
            return await self._deepseek_fim(prefix, suffix, stream)
        if self.cfg.fim_protocol == "openai_completions":
            return await self._openai_completions_fim(prefix, suffix, stream)
        if self.cfg.fim_protocol == "llamacpp_infill":
            return await self._llamacpp_infill(prefix, suffix, stream)
        raise ProviderError("config", f"unknown fim_protocol: {self.cfg.fim_protocol}")

    async def _deepseek_fim(self, prefix, suffix, stream):
        kwargs = dict(
            model=self.cfg.model,
            prompt=prefix,
            suffix=suffix,
            max_tokens=min(self.cfg.max_tokens, 4096),
            temperature=self.cfg.temperature,
        )
        if stream:
            return await self._stream_completions(kwargs)
        response = await self._with_retries(lambda: self.client.completions.create(**kwargs))
        usage = response.usage
        return {
            "text": response.choices[0].text or "",
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "live_cost": None,
            "ttft_ms": None,
        }

    async def _openai_completions_fim(self, prefix, suffix, stream):
        if self.cfg.fim_template:
            prompt = self.cfg.fim_template.replace("{prefix}", prefix).replace("{suffix}", suffix)
            kwargs = dict(
                model=self.cfg.model,
                prompt=prompt,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )
        else:
            kwargs = dict(
                model=self.cfg.model,
                prompt=prefix,
                suffix=suffix,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )
        if stream:
            return await self._stream_completions(kwargs)
        response = await self._with_retries(lambda: self.client.completions.create(**kwargs))
        usage = response.usage
        return {
            "text": response.choices[0].text or "",
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "live_cost": None,
            "ttft_ms": None,
        }

    async def _stream_completions(self, kwargs):
        start = time.perf_counter()
        ttft = None
        parts = []
        try:
            stream = await self.client.completions.create(**kwargs, stream=True)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].text:
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    parts.append(chunk.choices[0].text)
        except Exception as e:
            raise ProviderError(type(e).__name__, str(e))
        return {
            "text": "".join(parts),
            "prompt_tokens": None,
            "completion_tokens": None,
            "live_cost": None,
            "ttft_ms": ttft,
        }

    async def _llamacpp_infill(self, prefix, suffix, stream):
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
                    last_error = ProviderError("http", f"HTTP {response.status_code}: {response.text[:300]}")
                    await asyncio.sleep(BACKOFFS[attempt])
                    continue
                if response.status_code != 200:
                    raise ProviderError("http", f"HTTP {response.status_code}: {response.text[:300]}")
                data = response.json()
                content = data.get("content", "")
                return {
                    "text": content,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "live_cost": None,
                    "ttft_ms": (time.perf_counter() - start) * 1000,
                }
            except ProviderError:
                raise
            except Exception as e:
                last_error = ProviderError(type(e).__name__, str(e))
                await asyncio.sleep(BACKOFFS[attempt])
        raise last_error

    async def _with_retries(self, call):
        last_error = None
        for attempt in range(3):
            try:
                return await call()
            except Exception as e:
                status = getattr(e, "status_code", None)
                retryable = status in RETRYABLE_STATUS or isinstance(
                    e, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)
                )
                if not retryable or attempt == 2:
                    raise ProviderError(type(e).__name__, str(e))
                last_error = e
                await asyncio.sleep(BACKOFFS[attempt])
        raise ProviderError(type(last_error).__name__, str(last_error))

    async def close(self):
        await self.client.close()
