"""Prepare original InternVL inputs per row and batch the language model."""

import time


PREFILL_TOKEN_BUDGET = 2048


def build_requests(backend, jobs):
    from meme_study.freeze import _safe_file
    from explanation_study import RESERVED

    requests = []
    for job in jobs:
        messages = job['messages']
        blocks = messages[0]['content']
        images = [str(_safe_file(backend.assets, block['image']))
                  for block in blocks if block['type'] == 'image']
        if len(messages) != 1 or messages[0]['role'] != 'user' or len(images) != 1:
            raise ValueError('judge request requires one user message and one image')
        raw = '\n'.join(block['text'] for block in blocks if block['type'] == 'text')
        if any(marker in raw for marker in RESERVED):
            raise ValueError('judge request contains reserved model-template tokens')
        prompt = backend.template(backend.processor, backend.model.config, messages, num_images=1)
        requests.append((prompt, images))
    return requests


def generate_batch(backend, requests, generation):
    import mlx.core as mx
    from mlx_vlm.generate.ar import BatchGenerator
    from mlx_vlm.generate.dispatch import _prepare_generation_inputs

    model, processor = backend.model, backend.processor
    tokenizer = processor.tokenizer
    tokenizer.stopping_criteria.reset(model.config.eos_token_id)
    skip_ids = set(tokenizer.all_special_ids) if generation['skip_special_tokens'] else []
    if generation['temperature'] != 0:
        raise ValueError('this comparison requires greedy generation')
    max_tokens = generation['max_new_tokens']
    prompts, per_row_kwargs = [], []
    mx.reset_peak_memory()
    started = time.perf_counter()
    for prompt, images in requests:
        kwargs = {'temperature': generation['temperature']}
        ids, pixels, mask, _ = _prepare_generation_inputs(
            model, processor, prompt, images, None, None, kwargs
        )
        if ids.ndim != 2 or ids.shape[0] != 1 or pixels is None:
            raise ValueError('native preprocessing must return one multimodal row')
        if mask is not None and not all(mask[0].tolist()):
            raise ValueError('unexpected padding in a single-item native request')
        embedding = model.get_input_embeddings(ids, pixels, mask=mask, **kwargs)
        if embedding.inputs_embeds.shape[:2] != ids.shape:
            raise ValueError('native token IDs and embeddings disagree')
        mx.eval(embedding.inputs_embeds)
        prompts.append(ids[0].tolist())
        per_row_kwargs.append({k: v for k, v in embedding.to_dict().items() if v is not None})
    embedded_at = time.perf_counter()

    gen = BatchGenerator(
        model.language_model, processor, compute_logprobs=False,
        prefill_step_size=PREFILL_TOKEN_BUDGET,
        prefill_batch_size=len(requests), completion_batch_size=len(requests),
    )
    try:
        uids = gen.insert(prompts, max_tokens=max_tokens, prompt_kwargs=per_row_kwargs)
        token_ids = {uid: [] for uid in uids}
        reasons = {}
        finished_at = {}
        while gen.has_work:
            _, responses = gen.next()
            for response in responses:
                if response.token is not None:
                    token_ids[response.uid].append(int(response.token))
                if response.finish_reason is not None:
                    reasons[response.uid] = response.finish_reason
                    finished_at[response.uid] = time.perf_counter()
        batch_stats = gen.stats()
    finally:
        gen.close()

    results = []
    for uid, prompt in zip(uids, prompts):
        reason = reasons[uid]
        tokens = token_ids[uid]
        detokenizer = processor.detokenizer
        detokenizer.reset()
        segments = []
        for token in tokens[:-1] if reason == 'stop' else tokens:
            detokenizer.add_token(token, skip_special_token_ids=skip_ids)
            segments.append(detokenizer.last_segment)
        detokenizer.finalize()
        segments.append(detokenizer.last_segment)
        text = ''.join(segments)
        if callable(clean := getattr(processor, 'clean_output', None)):
            text = clean(text)
        results.append({
            'text': text, 'token_ids': tokens, 'input_tokens': len(prompt),
            'output_tokens': len(tokens), 'finish_reason': reason,
            'latency_s': finished_at[uid] - started,
        })
    return results, {
        'latency_s': time.perf_counter() - started,
        'embedding_latency_s': embedded_at - started,
        'language_prefill_latency_s': batch_stats.prompt_time,
        'peak_memory_bytes': mx.get_peak_memory(),
        'prefill_token_budget': PREFILL_TOKEN_BUDGET,
    }
