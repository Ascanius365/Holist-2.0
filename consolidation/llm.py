import re
import json
import inspect
import warnings
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from litellm import acompletion, ModelResponse
from dotenv import load_dotenv
from consolidation.io import read_yaml
import tiktoken
from transformers import AutoTokenizer

import os
from dotenv import load_dotenv

# Environment-Variablen laden
load_dotenv()


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self):
        return asdict(self)


@dataclass
class LLMResponse:
    response: str
    model: str
    usage: Usage

    def to_dict(self):
        return asdict(self)


class LLMAgent:
    def __init__(self, model_name: str):
        """Initialize LLM client"""
        self._model_name = model_name
        self._valid_parameter = set(inspect.signature(acompletion).parameters.keys()) | {
            "api_base",
            "num_retries",
            "custom_llm_provider"
        }
        # Fallback-Tokenizer (wird je nach Bedarf initialisiert)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    async def get_completion(
            self,
            prompt_path: Optional[str] = None,
            messages: Dict[str, Any] = None,
            api_base: Optional[str] = "https://openrouter.ai/api/v1",
            raw_response: bool = False,
            **kwargs,
    ) -> LLMResponse | ModelResponse:
        """Send messages to LLM via OpenRouter and get completion response"""
        if prompt_path is None and messages is None:
            raise ValueError("Either prompt_path or messages must be provided")
        elif prompt_path is not None and messages is not None:
            raise ValueError("prompt_path and messages cannot both be provided")
        elif messages is not None:
            prompt = {"messages": messages}
        elif prompt_path is not None:
            prompt = read_yaml(prompt_path)
        messages = self._make_messages(prompt, **kwargs)

        # Sicherstellen, dass das Modell das korrekte OpenRouter-Präfix besitzt
        target_model = self._model_name
        if not target_model.startswith("openrouter/"):
            target_model = f"openrouter/{target_model}"

        messages["model"] = target_model
        messages["api_base"] = api_base or "https://openrouter.ai/api/v1"
        messages["custom_llm_provider"] = "openrouter"

        # Absichern der validen Parameter für LiteLLM acompletion
        payload = {k: v for k, v in messages.items() if k in self._valid_parameter}

        payload["response_format"] = {"type": "json_object"}

        """
        # ==================== NEU: PROMPT-TEXT AUSGEBEN ====================
        print("\n" + "=" * 40 + " GESENDETER PROMPT " + "=" * 40)
        for msg in payload.get("messages", []):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            print(f"[{role}]:\n{content}\n")
        print("=" * 99 + "\n")
        # ===================================================================
        """

        # Generate response through LiteLLM
        response = await acompletion(**payload)

        if raw_response:
            return response

        return LLMResponse(
            response=response.choices[0].message.content,
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    def _make_messages(self, prompt, **kwargs):
        messages = self._fill_prompt(prompt, **kwargs)
        for key in list(self._valid_parameter):
            if key in kwargs:
                messages[key] = kwargs[key]
        return messages

    def _fill_prompt(self, prompt, **kwargs):
        def _fill(msg, kwargs):
            all_context_input = []
            context_input = re.findall(r"(\{\{\$.+?\}\})", msg)
            for input_ in context_input:
                str_to_replace = kwargs[input_[3:-2]]
                all_context_input.append(input_[3:-2])
                if isinstance(str_to_replace, int):
                    str_to_replace = str(str_to_replace)
                if isinstance(str_to_replace, list):
                    str_to_replace = "- " + "\n- ".join(str_to_replace)
                if isinstance(str_to_replace, dict):
                    str_to_replace = json.dumps(str_to_replace).replace('", "', '",\n"')
                msg = msg.replace(input_, str_to_replace)
            return msg, all_context_input

        prompt = deepcopy(prompt)
        all_context_input = []

        for msg in prompt["messages"]:
            output = _fill(msg["content"], kwargs)
            msg["content"] = output[0]
            all_context_input.extend(output[1])
        if len(set(kwargs) - (set(all_context_input) | self._valid_parameter)) > 0:
            raise ValueError(
                f"Invalid context input: {set(kwargs) - (set(all_context_input) | self._valid_parameter)}"
            )
        return prompt

    def count_tokens(self, text: Union[str, List[Dict[str, str]]]) -> int:
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
            combined_text = ""
            for msg in text:
                if "role" in msg and "content" in msg:
                    combined_text += f"{msg['role']}: {msg['content']}\n"
            text = combined_text
        return len(self._tokenizer.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        return self._tokenizer.decode(self._tokenizer.encode(text)[:max_tokens])