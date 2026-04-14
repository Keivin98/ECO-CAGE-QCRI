# from tqdm import tqdm
# import numpy as np
# from transformers import AutoTokenizer
# from datasets import load_dataset
# import os


# hf_tknzr = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


# def get_c4_data(datasets_dir, num_proc=40):
#     C4_DATA_PATH = os.path.join(datasets_dir, "c4/")
#     HF_CACHE_DIR = os.path.join(datasets_dir, ".hf_cache")

#     if not os.path.exists(os.path.join(C4_DATA_PATH, "train.bin")):
#         os.makedirs(C4_DATA_PATH, exist_ok=True)
#         os.makedirs(HF_CACHE_DIR, exist_ok=True)
#         dataset = load_dataset("allenai/c4", "en", cache_dir=HF_CACHE_DIR)

#         split_dataset = dataset["train"].train_test_split(
#             test_size=0.0005, seed=2357, shuffle=True
#         )
#         split_dataset["val"] = split_dataset.pop("test")

#         def process(example):
#             ids = hf_tknzr.encode(
#                 text=example["text"],
#                 add_special_tokens=True,
#                 padding=False,
#                 truncation=False,
#             )
#             out = {"ids": ids, "len": len(ids)}
#             return out

#         # tokenize the dataset
#         tokenized = split_dataset.map(
#             process,
#             remove_columns=["text"],
#             desc="tokenizing the splits",
#             num_proc=num_proc,
#             cache_file_name=None,
#         )

#         # concatenate all the ids in each dataset into one large file we can use for training
#         for split, dset in tokenized.items():
#             arr_len = np.sum(dset["len"])
#             filename = os.path.join(C4_DATA_PATH, f"{split}.bin")
#             dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
#             arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
#             total_batches = min(1024, len(dset))

#             idx = 0
#             for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
#                 # Batch together samples for faster write
#                 batch = dset.shard(
#                     num_shards=total_batches, index=batch_idx, contiguous=True
#                 ).with_format("numpy")
#                 arr_batch = np.concatenate(batch["ids"])
#                 # Write into mmap
#                 arr[idx : idx + len(arr_batch)] = arr_batch
#                 idx += len(arr_batch)
#             arr.flush()

#     return {
#         "train": os.path.join(C4_DATA_PATH, "train.bin"),
#         "val": os.path.join(C4_DATA_PATH, "val.bin"),
#     }


from __future__ import annotations

import os
import time
import random
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

hf_tknzr = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)


def _get_rank() -> int:
    # Works for torchrun; safe default for single-process
    return int(os.environ.get("RANK", "0"))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


@contextmanager
def _file_lock(lock_path: str, poll_s: float = 1.0) -> Iterator[None]:
    """
    Very simple lock using atomic create of a lock file.
    Works on shared filesystems for typical cases.
    """
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(poll_s)
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _encode_to_uint16(text: str) -> np.ndarray:
    ids = hf_tknzr.encode(
        text=text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    # LLaMA2 vocab (32k) fits in uint16
    return np.asarray(ids, dtype=np.uint16)


def _iter_stream_examples(stream: Iterable[dict], seed: int, buffer_size: int) -> Iterator[dict]:
    """
    Approximate shuffle for streaming datasets using a bounded buffer.
    """
    rng = random.Random(seed)
    buf = []
    it = iter(stream)

    # Fill buffer
    for _ in range(buffer_size):
        try:
            buf.append(next(it))
        except StopIteration:
            break

    # Pop/replace
    while buf:
        j = rng.randrange(len(buf))
        yield buf[j]
        try:
            buf[j] = next(it)
        except StopIteration:
            buf.pop(j)


def _write_token_stream(
    ex_iter: Iterable[dict],
    out_path: str,
    target_tokens: int,
    desc: str,
    *,
    enable_pbar: bool,
    batch_size: int = 256,            # tune: 128/256/512 depending on RAM
    write_chunk_tokens: int = 8_000_000,  # ~16MB per flush (uint16)
) -> int:
    """
    Faster version: batch tokenize + chunked writes.
    Output is identical to per-example encode+tofile, as long as:
      - ex_iter yields examples in the same order
      - tokenizer settings are the same
    """
    if isinstance(target_tokens, (tuple, list)):
        raise TypeError(f"target_tokens must be int, got {type(target_tokens)}: {target_tokens}")
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0, got {target_tokens}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    written = 0
    pbar = tqdm(total=target_tokens, desc=desc, unit="tok", disable=not enable_pbar)

    # Accumulate token arrays to write in large contiguous chunks
    chunk_parts: list[np.ndarray] = []
    chunk_tok = 0

    def flush_chunk(f) -> None:
        nonlocal chunk_parts, chunk_tok, written
        if not chunk_parts:
            return

        arr = np.concatenate(chunk_parts).astype(np.uint16, copy=False)

        remaining = target_tokens - written
        if arr.size > remaining:
            arr = arr[:remaining]

        arr.tofile(f)
        written += int(arr.size)
        pbar.update(int(arr.size))

        chunk_parts = []
        chunk_tok = 0

    with open(out_path, "wb") as f:
        batch_texts: list[str] = []

        for ex in ex_iter:
            if written >= target_tokens:
                break

            batch_texts.append(ex["text"])
            if len(batch_texts) < batch_size:
                continue

            enc = hf_tknzr(
                batch_texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )

            # Preserve ordering: iterate enc["input_ids"] in the same order as batch_texts
            for ids in enc["input_ids"]:
                arr = np.asarray(ids, dtype=np.uint16)
                chunk_parts.append(arr)
                chunk_tok += int(arr.size)

                # Flush if chunk is big or we're about to exceed target
                if chunk_tok >= write_chunk_tokens or (written + chunk_tok) >= target_tokens:
                    flush_chunk(f)
                    if written >= target_tokens:
                        break

            batch_texts = []

        # Handle any leftover texts
        if written < target_tokens and batch_texts:
            enc = hf_tknzr(
                batch_texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            for ids in enc["input_ids"]:
                arr = np.asarray(ids, dtype=np.uint16)
                chunk_parts.append(arr)
                chunk_tok += int(arr.size)
                if chunk_tok >= write_chunk_tokens or (written + chunk_tok) >= target_tokens:
                    flush_chunk(f)
                    if written >= target_tokens:
                        break

        # Final flush
        if written < target_tokens:
            flush_chunk(f)

    pbar.close()

    if written < target_tokens:
        raise RuntimeError(
            f"Stream ended early: wrote {written:,} tokens but target is {target_tokens:,}."
        )
    return written


def get_c4_data(
    datasets_dir: str,
    num_proc: int = 128,
    *,
    train_tokens: int = 20_000_000_000,
    val_tokens: int = 50_000_000,
    seed: int = 2357,
    shuffle_buffer: int = 100_000,
) -> Dict[str, str]:
    C4_DATA_PATH = os.path.join(datasets_dir, "c4")
    train_path = os.path.join(C4_DATA_PATH, "train.bin")
    val_path = os.path.join(C4_DATA_PATH, "val.bin")

    # Atomic-build paths
    train_tmp = train_path + ".tmp"
    val_tmp = val_path + ".tmp"
    done_path = os.path.join(C4_DATA_PATH, ".build.done")
    lock_path = os.path.join(C4_DATA_PATH, ".build.lock")

    rank = _get_rank()
    world = _get_world_size()

    os.makedirs(C4_DATA_PATH, exist_ok=True)

    def is_ready() -> bool:
        # Require marker + non-empty files
        return (
            os.path.exists(done_path)
            and os.path.exists(train_path) and os.path.getsize(train_path) > 0
            and os.path.exists(val_path) and os.path.getsize(val_path) > 0
        )

    # Fast path: already built safely
    if is_ready():
        return {"train": train_path, "val": val_path}

    if rank == 0:
        with _file_lock(lock_path):
            # Re-check inside lock
            if not is_ready():
                # Clean up any partial artifacts from previous failed runs
                for p in [train_tmp, val_tmp, done_path]:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

                # IMPORTANT: write to tmp files, then rename atomically
                stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
                stream = _iter_stream_examples(stream, seed=seed, buffer_size=shuffle_buffer)

                _write_token_stream(
                    stream,
                    val_tmp,
                    val_tokens,
                    f"writing {os.path.basename(val_path)}",
                    enable_pbar=True,
                )
                os.replace(val_tmp, val_path)   # atomic

                _write_token_stream(
                    stream,
                    train_tmp,
                    train_tokens,
                    f"writing {os.path.basename(train_path)}",
                    enable_pbar=True,
                )
                os.replace(train_tmp, train_path)  # atomic

                # Write a done marker last
                with open(done_path, "w") as f:
                    f.write("ok\n")

    # Everyone waits until done marker exists and files are non-empty
    if world > 1:
        while not is_ready():
            time.sleep(2.0)

    return {"train": train_path, "val": val_path}
