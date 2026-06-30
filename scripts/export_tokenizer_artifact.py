#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

from gguf import GGUFReader


def field_string(reader: GGUFReader, name: str) -> str:
    field = reader.fields[name]
    value = field.parts[-1]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tobytes"):
        return value.tobytes().decode("utf-8")
    return str(value)


def write_string(out, value: str) -> None:
    data = value.encode("utf-8")
    out.write(struct.pack("<I", len(data)))
    out.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default="/root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.gguf")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reader = GGUFReader(args.gguf)
    tokenizer = json.loads(field_string(reader, "tokenizer.huggingface.json"))
    chat_template = field_string(reader, "tokenizer.chat_template")
    vocab_size = int(reader.fields["qwen3.vocab_size"].parts[-1][0])

    vocab_items = sorted(tokenizer["model"]["vocab"].items(), key=lambda item: item[1])
    added_tokens = sorted(tokenizer.get("added_tokens", []), key=lambda item: item["id"])
    max_id = max([int(token_id) for _, token_id in vocab_items] + [int(token["id"]) for token in added_tokens])
    tokens = [""] * max(vocab_size, max_id + 1)
    token_types = [1] * len(tokens)
    for token, token_id in vocab_items:
        tokens[int(token_id)] = token
    for token in added_tokens:
        token_id = int(token["id"])
        tokens[token_id] = token["content"]
        token_types[token_id] = 3 if token.get("special", False) else 4

    merges = [" ".join(pair) for pair in tokenizer["model"]["merges"]]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        out.write(b"HATK1\0\0\0")
        out.write(struct.pack("<IIII", len(tokens), len(merges), len(added_tokens), len(chat_template.encode("utf-8"))))
        for token in tokens:
            write_string(out, token)
        for token_type in token_types:
            out.write(struct.pack("<i", token_type))
        for merge in merges:
            write_string(out, merge)
        write_string(out, chat_template)

    print("wrote", out_path, "tokens", len(tokens), "merges", len(merges), "added", len(added_tokens))


if __name__ == "__main__":
    main()
