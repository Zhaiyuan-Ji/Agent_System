from __future__ import annotations

from .runtime_models import ContextBlock


def assemble_context_blocks(
    blocks: list[ContextBlock],
    token_budget: int,
) -> tuple[list[ContextBlock], list[dict[str, str]]]:
    selected: list[ContextBlock] = []
    dropped: list[dict[str, str]] = []
    used_tokens = 0

    for block in sorted(blocks, key=lambda item: item.priority, reverse=True):
        estimated_tokens = max(block.estimated_tokens, 0)
        if used_tokens + estimated_tokens <= token_budget:
            selected.append(
                ContextBlock(
                    **{
                        **block.to_dict(),
                        "selected": True,
                        "drop_reason": None,
                    }
                )
            )
            used_tokens += estimated_tokens
        else:
            dropped.append({"block_id": block.block_id, "reason": "budget"})

    return selected, dropped
