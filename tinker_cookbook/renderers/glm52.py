"""
GLM-5.2 renderer (zai-org/GLM-5.2, zai-org/GLM-5.2-FP8).

Reproduces the model's HF ``chat_template.jinja`` for the chat/RL path so the
prompt tokens match ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``
exactly (validated token-for-token against the GLM-5.2 tokenizer).

Format (thinking enabled, the default), no separating newlines:

    [gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{question}<|assistant|><think>

The model then samples ``{reasoning}</think>{answer}`` and stops on one of
``<|endoftext|>`` / ``<|user|>`` / ``<|observation|>``. GLM has no per-message
end token; a turn ends when the next role marker (or EOS) appears.

Key GLM-5.2 specifics vs the GLM-5/5.1 family (see prime-rl's ``glm5.py``):
- a leading ``<|system|>Reasoning Effort: Max`` line is injected whenever
  thinking is enabled (GLM-5/5.1 do not emit this);
- structural markers carry no separating newlines (GLM-4.5/4.6 did);
- the generation prompt ends with a dangling open ``<think>`` (thinking on) or
  ``<think></think>`` (thinking off).

Token ids (GLM-5.2 tokenizer): [gMASK]=154822, <sop>=154824, <|system|>=154826,
<|user|>=154827, <|assistant|>=154828, <|observation|>=154829, <think>=154841,
</think>=154842, <|endoftext|>=154820.
"""

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ThinkingPart,
    TextPart,
    parse_content_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_ROLE_MARKER = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|observation|>",
}

_STOP_TOKENS = ("<|endoftext|>", "<|user|>", "<|observation|>")


def _text_of(content: object) -> str:
    """Flatten message content (string or list of parts) to visible text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p["text"] for p in content if p.get("type") == "text")
    return str(content)


def _reasoning_of(content: object) -> str:
    if isinstance(content, list):
        return "".join(p["thinking"] for p in content if p.get("type") == "thinking")
    return ""


class GLM52Renderer(Renderer):
    """Renderer for GLM-5.2 with thinking enabled (HF template default)."""

    supports_streaming = False

    def __init__(
        self,
        tokenizer: Tokenizer,
        enable_thinking: bool = True,
        reasoning_effort: str = "max",
    ):
        super().__init__(tokenizer)
        self.enable_thinking = enable_thinking
        # HF template: 'high' -> "High"; anything else -> "Max".
        self.reasoning_effort = "high" if reasoning_effort == "high" else "max"

    def _encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    @property
    def _bos_tokens(self) -> list[int]:
        # [gMASK]<sop> sequence scaffold, plus the Reasoning-Effort system line
        # that GLM-5.2 injects whenever thinking is enabled.
        prefix = "[gMASK]<sop>"
        if self.enable_thinking:
            effort = "High" if self.reasoning_effort == "high" else "Max"
            prefix += f"<|system|>Reasoning Effort: {effort}"
        return self._encode(prefix)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        content = message["content"]

        if role == "assistant":
            header_str = "<|assistant|>"
            reasoning = _reasoning_of(content)
            text = _text_of(content)
            # Current/last assistant turn keeps its reasoning; historical turns
            # collapse to an empty think block (matches the template's else-branch).
            keep_reasoning = ctx.idx > ctx.last_user_index
            if reasoning and keep_reasoning:
                body = f"<think>{reasoning}</think>{text}"
            else:
                body = f"<think></think>{text}"
            output_str = body
        elif role == "tool":
            # New <|observation|> marker only when the previous message wasn't a tool.
            prev = ctx.prev_message
            open_obs = prev is None or prev.get("role") != "tool"
            header_str = "<|observation|>" if open_obs else ""
            output_str = f"<tool_response>{_text_of(content)}</tool_response>"
        else:
            # system / user
            header_str = _ROLE_MARKER[role]
            output_str = _text_of(content)

        header = tinker.types.EncodedTextChunk(tokens=self._encode(header_str))
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(tokens=self._encode(output_str))
        ]
        return RenderedMessage(header=header, output=output)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        if role == "assistant":
            # Thinking on -> dangling open <think>; off -> closed empty block.
            suffix = "<|assistant|>" + ("<think>" if self.enable_thinking else "<think></think>")
            return self._encode(suffix)
        return self._encode(_ROLE_MARKER.get(role, "<|assistant|>"))

    def get_stop_sequences(self) -> list[int]:
        stops: list[int] = []
        for tok in _STOP_TOKENS:
            ids = self._encode(tok)
            assert len(ids) == 1, f"Expected single token for {tok!r}, got {ids}"
            stops.append(ids[0])
        return stops

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        response = self._normalize_response_tokens(response)
        stops = set(self.get_stop_sequences())
        eos = self._encode("<|endoftext|>")[0]

        stop_idx = next((i for i, t in enumerate(response) if t in stops), None)
        if stop_idx is None:
            content_tokens = response
            termination = ParseTermination.MALFORMED
        else:
            content_tokens = response[:stop_idx]
            termination = (
                ParseTermination.EOS
                if response[stop_idx] == eos
                else ParseTermination.STOP_SEQUENCE
            )

        text = str(self.tokenizer.decode(content_tokens))
        # The generation prompt already emitted the opening <think>; re-add it so
        # the shared block parser sees a well-formed <think>...</think>{answer}.
        reconstructed = "<think>" + text

        result = parse_content_blocks(reconstructed)
        message = Message(role="assistant", content=reconstructed)
        if result is not None:
            parts, _tool_results = result
            message["content"] = parts
        return message, termination
