import json
import os
import re
from urllib.parse import urlparse

PROTOCOLS = ("deepseek_fim", "openai_completions", "llamacpp_infill")
ENV_VAR = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


class ConfigError(Exception):
    pass


def load_config(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"invalid JSON in {path}: {e}")
    if not isinstance(data, list) or not data:
        raise ConfigError("config must be a non-empty JSON array")
    models = [validate_model(entry, i) for i, entry in enumerate(data)]
    names = [m.name for m in models]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"duplicate model names: {', '.join(duplicates)}")
    return models


def active_models(models):
    return [model for model in models if model.is_active()]


def is_local_url(url):
    if not url:
        return False
    return urlparse(url).hostname in LOCAL_HOSTS


def validate_model(entry, index):
    if not isinstance(entry, dict):
        raise ConfigError(f"model #{index}: entry must be an object")
    errors = []
    name = entry.get("name") or entry.get("model") or f"model-{index}"
    if not isinstance(name, str) or not name.strip():
        errors.append("name: required string")
    model = entry.get("model")
    if not isinstance(model, str) or not model.strip():
        errors.append("model: required string")
    endpoint = entry.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
        errors.append("endpoint: required http(s) URL")
    fim_endpoint = entry.get("fim_endpoint")
    fim_protocol = entry.get("fim_protocol")
    if fim_endpoint is not None and not isinstance(fim_endpoint, str):
        errors.append("fim_endpoint: must be a URL string")
    if fim_protocol is not None and fim_protocol not in PROTOCOLS:
        errors.append(f"fim_protocol: must be one of {PROTOCOLS}")
    if (fim_endpoint is None) != (fim_protocol is None):
        errors.append("fim_endpoint and fim_protocol must be set together")
    template = entry.get("fim_template")
    if template is not None and not isinstance(template, str):
        errors.append("fim_template: must be a string")
    max_context = entry.get("max_context_size")
    if max_context is not None and (not isinstance(max_context, int) or max_context <= 0):
        errors.append("max_context_size: must be a positive integer")
    max_tokens = entry.get("max_tokens", 256)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        errors.append("max_tokens: must be a positive integer")
    temperature = entry.get("temperature", 0)
    if not isinstance(temperature, (int, float)):
        errors.append("temperature: must be a number")
    timeout = entry.get("timeout")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        errors.append("timeout: must be a positive number")
    headers = entry.get("headers", {})
    if not isinstance(headers, dict):
        errors.append("headers: must be an object")
    extra_body = entry.get("extra_body", {})
    if not isinstance(extra_body, dict):
        errors.append("extra_body: must be an object")
    modes = entry.get("modes", ["fim", "chat"])
    if (not isinstance(modes, list) or not modes
            or any(m not in ("fim", "chat") for m in modes)
            or len(set(modes)) != len(modes)):
        errors.append("modes: must be a non-empty list of unique fim/chat")
    device_type = entry.get("device_type")
    if device_type is not None and (not isinstance(device_type, str) or not device_type.strip()):
        errors.append("device_type: must be a non-empty string")
    deactivated = entry.get("deactivated")
    if deactivated is not None:
        if not isinstance(deactivated, dict):
            errors.append("deactivated: must be an object")
        elif deactivated.get("reason") is not None and not isinstance(deactivated["reason"], str):
            errors.append("deactivated: reason must be a string")
    pricing = entry.get("pricing")
    if pricing is not None:
        if not isinstance(pricing, dict) or not all(
            isinstance(pricing.get(k), (int, float)) for k in ("input", "output")
        ):
            errors.append("pricing: must be {input: number, output: number}")
    key_env = entry.get("api_key_env")
    key = entry.get("api_key")
    if key_env is not None and not isinstance(key_env, str):
        errors.append("api_key_env: must be a string or null")
    if key is not None and not isinstance(key, str):
        errors.append("api_key: must be a string or null")
    if errors:
        raise ConfigError(f"model #{index} ({name}): " + "; ".join(errors))
    return ModelConfig(
        name=name.strip(),
        model=model.strip(),
        endpoint=endpoint,
        fim_endpoint=fim_endpoint,
        fim_protocol=fim_protocol,
        fim_template=template,
        api_key=resolve_key(key, key_env),
        max_context_size=max_context,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        headers=headers,
        extra_body=extra_body,
        modes=modes,
        pricing=pricing,
        device_type=device_type.strip() if device_type else None,
        deactivated=deactivated,
    )


def resolve_key(key, key_env):
    if key_env:
        value = os.environ.get(key_env)
        if value is None:
            raise ConfigError(f"api_key_env: environment variable {key_env} is not set")
        return value
    if key and ENV_VAR.match(key):
        name = ENV_VAR.match(key).group(1)
        value = os.environ.get(name)
        if value is None:
            raise ConfigError(f"api_key: environment variable {name} is not set")
        return value
    return key


class ModelConfig:
    def __init__(self, name, model, endpoint, fim_endpoint, fim_protocol, fim_template,
                 api_key, max_context_size, max_tokens, temperature, timeout, headers,
                 extra_body, modes, pricing, device_type, deactivated):
        self.name = name
        self.model = model
        self.endpoint = endpoint
        self.fim_endpoint = fim_endpoint
        self.fim_protocol = fim_protocol
        self.fim_template = fim_template
        self.api_key = api_key
        self.max_context_size = max_context_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.headers = headers
        self.extra_body = extra_body
        self.modes = modes
        self.pricing = pricing
        self.device_type = device_type
        self.deactivated = deactivated

    def has_fim(self):
        return self.fim_endpoint is not None

    def is_active(self):
        return self.deactivated is None

    def is_local(self):
        return is_local_url(self.endpoint) or is_local_url(self.fim_endpoint)
