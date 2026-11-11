"""EOS-aware native generation, with matched rolling summary windows."""
from dataclasses import fields, is_dataclass
import time

import torch


def select_cache(value, indices):
    if torch.is_tensor(value):
        return value.index_select(0, indices) if value.ndim else value
    if is_dataclass(value):
        return type(value)(
            **{f.name: select_cache(getattr(value, f.name), indices) for f in fields(value)}
        )
    if isinstance(value, tuple):
        return tuple(select_cache(x, indices) for x in value)
    if isinstance(value, list):
        return [select_cache(x, indices) for x in value]
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise TypeError("Unrecognized native cache state: " + str(type(value)))


@torch.inference_mode()
def generate_batch(model, sources, formatter, policy, device, autocast=True):
    if not sources:
        raise ValueError("Cannot generate an empty batch")
    model.eval()
    backbone = model.backbone
    tok = formatter.tokenizer
    generated = [[] for _ in sources]
    visible = [False] * len(sources)
    stop = [None] * len(sources)
    resets = [0] * len(sources)
    tables = [{} for _ in sources]
    active = list(range(len(sources)))
    offset = 0
    start_time = time.monotonic()
    limit = policy["max_new_tokens"]
    ngram = policy["no_repeat_ngram"]
    if limit < 1 or ngram < 1:
        raise ValueError("Invalid decoding guard")
    from contextlib import nullcontext

    def amp():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if autocast and torch.device(device).type == "cuda"
            else nullcontext()
        )

    while active and offset < limit:
        prefixes = [
            formatter.prefix(sources[i], generated[i][-formatter.s["history_tokens"] :], offset)
            for i in active
        ]
        width = max(map(len, prefixes))
        count = len(active)
        ids = torch.full((count, width), tok.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((count, width), dtype=torch.bool, device=device)
        for row, prefix in enumerate(prefixes):
            ids[row, : len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=device)
            mask[row, : len(prefix)] = True
        if width + min(formatter.s["block_tokens"], limit - offset) > formatter.s["context_tokens"]:
            raise ValueError("Rolling physical cache would exceed the original context")
        # Reset ALL mixed attention/recurrent/convolution state before re-prefill.
        with amp():
            out = backbone(
                ids,
                attention_mask=None if bool(mask.all()) else mask,
                use_cache=True,
                last_token_only=True,
            )
        cache = out.past_key_values
        logits = out.logits[:, 0]
        for i in active:
            resets[i] += 1
        steps = min(formatter.s["block_tokens"], limit - offset)
        for local_step in range(steps):
            scores = logits.float().clone()
            scores[:, tok.pad_id] = float("-inf")
            scores[:, tok.bos_id] = float("-inf")
            for row, i in enumerate(active):
                if not visible[i]:
                    scores[row, tok.eos_id] = float("-inf")
                if len(generated[i]) >= ngram - 1:
                    key = tuple(generated[i][-(ngram - 1) :]) if ngram > 1 else ()
                    banned = tables[i].get(key)
                    if banned:
                        scores[row, list(banned)] = float("-inf")
            if not bool(torch.isfinite(scores.amax(-1)).all()):
                raise FloatingPointError("No finite allowed token")
            chosen = scores.argmax(-1).tolist()
            keep = []
            pending = []
            new_active = []
            for row, (i, token) in enumerate(zip(active, chosen)):
                if token == tok.eos_id:
                    stop[i] = "eos"
                    continue
                generated[i].append(token)
                if not visible[i]:
                    visible[i] = bool(tok.decode(generated[i]).strip())
                if len(generated[i]) >= ngram:
                    key = tuple(generated[i][-ngram:-1]) if ngram > 1 else ()
                    tables[i].setdefault(key, set()).add(token)
                if len(generated[i]) >= limit:
                    stop[i] = "max_new_tokens"
                    continue
                keep.append(row)
                pending.append(token)
                new_active.append(i)
            if not new_active:
                active = []
                break
            if len(keep) != len(active):
                cache = select_cache(cache, torch.tensor(keep, dtype=torch.long, device=device))
            active = new_active
            if local_step + 1 < steps:
                token_ids = torch.tensor(pending, dtype=torch.long, device=device)[:, None]
                with amp():
                    out = backbone(
                        token_ids, past_key_values=cache, use_cache=True, last_token_only=True
                    )
                cache = out.past_key_values
                logits = out.logits[:, 0]
        offset += steps
    result = []
    for i, tokens in enumerate(generated):
        text = tok.decode(tokens).strip()
        if not text or stop[i] is None:
            raise ValueError("Generation did not produce a valid terminated summary")
        result.append(
            {
                "summary": text,
                "token_ids": tokens,
                "generated_tokens": len(tokens),
                "stop": stop[i],
                "prefill_blocks": resets[i],
                "elapsed_batch_seconds": time.monotonic() - start_time,
            }
        )
    return result
