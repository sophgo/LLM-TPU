# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import numpy as np
import time

import chat
from transformers import AutoTokenizer

DEFAULT_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


class Qwen3Embedding():

    def __init__(self, args):
        self.device = [int(d) for d in args.devid.split(",")]
        self.dim = args.dim

        print("Load " + args.config_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.config_path, trust_remote_code=True, padding_side="left"
        )
        self.tokenizer.decode([0])

        self.model = chat.Qwen3Embedding()
        load_start = time.time()
        self.model.init(self.device, args.model_path)
        load_end = time.time()
        print(f"Load Time: {(load_end - load_start):.3f} s")
        print(f"SEQLEN: {self.model.SEQLEN}, Hidden: {self.model.HIDDEN_SIZE}, "
              f"Layers: {self.model.NUM_LAYERS}, Dynamic: {self.model.is_dynamic}")

        hidden = self.model.HIDDEN_SIZE
        if self.dim > hidden:
            print(f"Warning: --dim {self.dim} > hidden size {hidden}, "
                  f"truncation has no effect; using full {hidden} dims")
            self.dim = hidden
        elif self.dim < 32:
            print(f"Warning: --dim {self.dim} < 32, below the MRL supported range "
                  f"(32-{hidden}); results may degrade")

    def __del__(self):
        self.model.deinit()

    def encode(self, texts, instruction=""):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            full_text = instruction + text if instruction else text
            input_ids = self.tokenizer(
                full_text, truncation=True,
                max_length=self.model.SEQLEN,
            ).input_ids
            token_length = len(input_ids)
            if token_length == 0:
                embeddings.append(np.zeros(self.dim, dtype=np.float32))
                continue

            fwd_start = time.time()
            hidden_flat = self.model.forward(input_ids)
            fwd_end = time.time()

            hidden = np.array(hidden_flat, dtype=np.float32).reshape(
                token_length, self.model.HIDDEN_SIZE
            )

            # last-token pooling
            emb = hidden[token_length - 1]

            # MRL truncation
            if self.dim < self.model.HIDDEN_SIZE:
                emb = emb[:self.dim]

            # L2 normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            embeddings.append(emb)
            print(f"  [{token_length} tokens, "
                  f"{(fwd_end - fwd_start) * 1000:.1f} ms]")

        return embeddings

    def cosine_similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main(args):
    model = Qwen3Embedding(args)

    if args.query and args.documents:
        docs = args.documents
        instruction = args.instruction if args.instruction else DEFAULT_INSTRUCTION

        print(f"\nEncoding query (instruction: '{instruction[:40]}...'):")
        print(f"  Query: {args.query}")
        q_emb = model.encode([args.query], instruction=instruction)

        print(f"\nEncoding {len(docs)} documents (no instruction):")
        d_embs = model.encode(docs, instruction="")

        print(f"\nSimilarity (dim={model.dim}):")
        for i, doc in enumerate(docs):
            score = model.cosine_similarity(q_emb[0], d_embs[i])
            print(f"  [{score:.4f}] {doc[:80]}")
    else:
        # interactive mode
        print("\n=== Qwen3-Embedding Interactive ===")
        print("Commands:")
        print("  /q or /exit    — quit")
        print("  /dim <N>       — set output dim (current: {})".format(model.dim))
        print("  /nodim         — reset dim to full hidden size")
        print("  /sim <t1> ||| <t2>  — cosine similarity between two texts")
        print("  <text>         — encode text (with default instruction)")
        while True:
            text = input("\nInput: ").strip()
            if not text:
                continue
            if text in ("/q", "/exit", "/quit"):
                break
            if text.startswith("/dim "):
                try:
                    new_dim = int(text.split()[1])
                except ValueError:
                    print("Usage: /dim <N>")
                    continue
                hidden = model.model.HIDDEN_SIZE
                if new_dim > hidden:
                    print(f"Warning: dim {new_dim} > hidden size {hidden}, "
                          f"truncation has no effect; using full {hidden} dims")
                    new_dim = hidden
                elif new_dim < 32:
                    print(f"Warning: dim {new_dim} < 32, below the MRL supported range "
                          f"(32-{hidden}); results may degrade")
                model.dim = new_dim
                print(f"Dim set to {model.dim}")
                continue
            if text == "/nodim":
                model.dim = model.model.HIDDEN_SIZE
                print(f"Dim reset to {model.dim}")
                continue
            if text.startswith("/sim "):
                parts = text[5:].split("|||")
                if len(parts) != 2:
                    print("Usage: /sim <text1> ||| <text2>")
                    continue
                t1, t2 = parts[0].strip(), parts[1].strip()
                instruction = args.instruction if args.instruction else DEFAULT_INSTRUCTION
                embs = model.encode([t1, t2], instruction=instruction)
                score = model.cosine_similarity(embs[0], embs[1])
                print(f"Cosine similarity: {score:.4f}")
                continue

            instruction = args.instruction if args.instruction else DEFAULT_INSTRUCTION
            embs = model.encode([text], instruction=instruction)
            print(f"Embedding (first 5): {embs[0][:5]}")
            print(f"Norm: {np.linalg.norm(embs[0]):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the tokenizer config directory')
    parser.add_argument('-d', '--devid', type=str, default='0',
                        help='device ID to use')
    parser.add_argument('--dim', type=int, default=1024,
                        help='output embedding dimension (MRL truncation, default 1024)')
    parser.add_argument('--query', type=str, default=None,
                        help='query text for retrieval demo')
    parser.add_argument('--documents', type=str, nargs='+', default=None,
                        help='document texts for retrieval demo')
    parser.add_argument('--instruction', type=str, default=None,
                        help='custom instruction prefix (default: web search query)')
    # yapf: enable
    args = parser.parse_args()
    main(args)
