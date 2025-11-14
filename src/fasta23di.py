#!/usr/bin/env python

import argparse
import re
import sys
import os

import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_fasta(fpath):
    """
    Simple FASTA iterator. Yields (header, sequence) pairs.
    Header is the line after '>' without the '>' character.
    Sequence is concatenated, no whitespaces.
    """
    header = None
    chunks = []
    with open(fpath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def wrap_fasta_sequence(seq, width=80):
    """Wrap sequence string to specified line width."""
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def preprocess_sequence(seq):
    """
    Preprocess amino acid sequence for ProstT5 AA→3Di:
    - uppercase
    - replace rare/ambiguous residues U,Z,O,B with X
    - add spaces between residues
    - prepend <AA2fold> prefix
    """
    seq = seq.upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    seq_spaced = " ".join(list(seq))
    return "<AA2fold> " + seq_spaced


def load_model(model_path: str, checkpoint_path: str | None = None):
    """
    Load tokenizer and model.

    - `model_path` can be a HF hub name or a local directory with `save_pretrained()`.
    - If `checkpoint_path` is provided, it is loaded via torch.load():

        * If it's a dict -> assumed to be a state_dict, loaded into the base model.
        * Otherwise       -> assumed to be a full model object saved with torch.save(model, ...).

    Tokenizer is always loaded from `model_path`.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

    # Start from base HF model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Heuristic: dict -> state_dict, else -> full model
        if isinstance(ckpt, dict):
            # state_dict
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing or unexpected:
                print(
                    f"[warning] When loading state_dict from {checkpoint_path}, "
                    f"missing keys: {missing}, unexpected keys: {unexpected}",
                    file=sys.stderr,
                )
        else:
            # full model object
            model = ckpt

    # Move to device + set precision
    model = model.to(device)
    if device.type == "cpu":
        model.float()
    else:
        model.half()

    return tokenizer, model


def aa_to_3di_batch(tokenizer, model, seqs_raw, gen_kwargs):
    """
    Convert a batch of AA sequences (strings) to 3Di strings (no spaces),
    using ProstT5 AA→3Di translation.
    """
    if len(seqs_raw) == 0:
        return []

    # lengths before tokenization
    min_len = min(len(s) for s in seqs_raw)
    max_len = max(len(s) for s in seqs_raw)

    inputs = [preprocess_sequence(s) for s in seqs_raw]

    ids = tokenizer.batch_encode_plus(
        inputs,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    # override max_length / min_length per batch to be safe
    local_kwargs = dict(gen_kwargs)
    local_kwargs.setdefault("max_length", max_len)
    local_kwargs.setdefault("min_length", min_len)

    with torch.no_grad():
        translations = model.generate(
            ids.input_ids,
            attention_mask=ids.attention_mask,
            num_return_sequences=1,
            **local_kwargs,
        )

    decoded = tokenizer.batch_decode(translations, skip_special_tokens=True)

    # remove spaces between tokens: "a b c" -> "abc"
    structure_sequences = ["".join(ts.split(" ")) for ts in decoded]

    return structure_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Convert AA FASTA to ProstT5 (or fine-tuned variant) 3Di token FASTA."
    )
    parser.add_argument("input_fasta", help="Input FASTA file with amino acid sequences")
    parser.add_argument("output_fasta", help="Output FASTA file for 3Di tokens")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for model inference (default: 8)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic decoding (beam search, no sampling).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Rostlab/ProstT5",
        help=(
            "Path or HF hub name of the base model+tokenizer "
            "(default: Rostlab/ProstT5). "
            "Used as the base architecture and tokenizer."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a PyTorch checkpoint saved with torch.save(). "
            "If it is a dict, it is treated as state_dict and loaded into "
            "the base model from --model-path. Otherwise, it is treated as "
            "a full model object."
        ),
    )

    args = parser.parse_args()

    # Load tokenizer/model from hub or local path, optionally overriding with torch checkpoint
    tokenizer, model = load_model(args.model_path, args.checkpoint)

    if args.deterministic:
        # Deterministic config: no sampling, just beam search
        gen_kwargs_aa2fold = {
            "do_sample": False,
            "num_beams": 4,
            "early_stopping": True,
        }
    else:
        # Stochastic config (original settings)
        gen_kwargs_aa2fold = {
            "do_sample": True,
            "num_beams": 3,
            "top_p": 0.95,
            "temperature": 1.2,
            "top_k": 6,
            "repetition_penalty": 1.2,
            "early_stopping": True,
        }

    batch_headers = []
    batch_seqs = []

    with open(args.output_fasta, "w") as out_f:
        for header, seq in read_fasta(args.input_fasta):
            batch_headers.append(header)
            batch_seqs.append(seq)

            if len(batch_seqs) >= args.batch_size:
                struct_seqs = aa_to_3di_batch(
                    tokenizer, model, batch_seqs, gen_kwargs_aa2fold
                )
                for h, s3di in zip(batch_headers, struct_seqs):
                    out_f.write(f">{h}\n")
                    out_f.write(wrap_fasta_sequence(s3di) + "\n")
                batch_headers = []
                batch_seqs = []

        # flush remaining
        if batch_seqs:
            struct_seqs = aa_to_3di_batch(
                tokenizer, model, batch_seqs, gen_kwargs_aa2fold
            )
            for h, s3di in zip(batch_headers, struct_seqs):
                out_f.write(f">{h}\n")
                out_f.write(wrap_fasta_sequence(s3di) + "\n")


if __name__ == "__main__":
    # Examples:
    #
    # 1) Baseline ProstT5, stochastic:
    #    python aa2fold_fasta.py input.fa output_3di.fa
    #
    # 2) Baseline ProstT5, deterministic:
    #    python aa2fold_fasta.py input.fa output_3di.fa --deterministic
    #
    # 3) Fine-tuned state_dict:
    #    torch.save(model.state_dict(), "ft_state_dict.pt")
    #    python aa2fold_fasta.py input.fa output_3di.fa \
    #        --model-path Rostlab/ProstT5 \
    #        --checkpoint ft_state_dict.pt --deterministic
    #
    # 4) Fine-tuned full model:
    #    torch.save(model, "ft_full_model.pt")
    #    python aa2fold_fasta.py input.fa output_3di.fa \
    #        --model-path Rostlab/ProstT5 \
    #        --checkpoint ft_full_model.pt --deterministic
    main()
