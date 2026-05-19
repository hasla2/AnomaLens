# pip install llama-cpp-python llama-index-llms-llama-cpp
import json
import re

from llama_cpp import Llama

SYSTEM_PROMPT = """You are an infrastructure log analyst for VMware ESXi and Hyper-V environments.
Analyze the given log template and classify it. Respond ONLY with valid JSON, no other text.

Labels:
- "noise": routine operational messages, heartbeats, API polling
- "important": VM/host lifecycle events, configuration changes, migrations
- "warning": performance degradation, retries, connectivity issues, resource pressure
- "error": failures, exceptions, authentication errors, timeouts
- "security_noise": login/logout events, session tracking, certificate checks
- "anomaly": anything that doesn't fit known patterns or looks suspicious

JSON schema:
{
  "label": "<one of the labels above>",
  "confidence": <float 0.0-1.0>,
  "explanation": "<one sentence in English>",
  "severity": <int 0-10>
}"""

class LLMDetector:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,   # CPU only
            verbose=False,
        )

    def classify(self, template: str, context_templates: list[str] = None) -> dict:
        """
        context_templates: последние N шаблонов из этого же окна (для контекста)
        """
        context_str = ""
        if context_templates:
            context_str = "\nRecent log context (last 5):\n" + \
                "\n".join(f"  - {t}" for t in context_templates[-5:])

        user_msg = f"Log template to classify:{context_str}\n\nTemplate: {template}"

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.1,   # детерминированность важнее creativity
            response_format={"type": "json_object"},
        )

        raw = response["choices"][0]["message"]["content"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # fallback: вытащить JSON из текста
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"label": "unknown", "confidence": 0.0,
                    "explanation": raw[:100], "severity": 0}

    def classify_window(self, templates: list[str]) -> list[dict]:
        """Классификация окна из 15 минут пакетом"""
        results = []
        for i, tpl in enumerate(templates):
            context = templates[max(0, i-5):i]
            result = self.classify(tpl, context)
            result["template"] = tpl
            results.append(result)
        return results