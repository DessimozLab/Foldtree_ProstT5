#!/usr/bin/env python

import argparse
import re
import sys
import os
import copy
import numpy as np

import torch
import torch.nn as nn
from transformers import (
    T5Tokenizer, 
    T5EncoderModel, 
    T5Config,
    T5PreTrainedModel
)
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class ClassConfig:
    def __init__(self, dropout=0.2, num_labels=20):
        self.dropout_rate = dropout
        self.num_labels = num_labels

class T5EncoderForTokenClassification(T5PreTrainedModel):
    def __init__(self, config: T5Config, class_config):
        super().__init__(config)
        self.num_labels = class_config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5EncoderModel(encoder_config)#, self.shared)

        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, class_config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # Convert the tensor_half to torch.float32 before the operation
        sequence_output_fl = sequence_output.to(torch.float32)
        logits = self.classifier(sequence_output_fl)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)

            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
            )

            valid_logits = active_logits[active_labels != -100]
            valid_labels = active_labels[active_labels != -100]

            valid_labels = valid_labels.type(torch.LongTensor).to(device)

            loss = loss_fct(valid_logits, valid_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def load_token_classification_model(model_path: str, checkpoint_path: str | None = None, half_precision=False):
    """
    Load ProstT5 as a token classification model for 3Di prediction.
    
    Args:
        model_path: Path or HF hub name for the base ProstT5 model
        checkpoint_path: Optional path to fine-tuned weights
        half_precision: Whether to use half precision
    
    Returns:
        tuple: (tokenizer, model)
    """
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    
    # Load base encoder model 
    if half_precision and torch.cuda.is_available():
        base_model = T5EncoderModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )
    else:
        base_model = T5EncoderModel.from_pretrained(model_path)
    
    # Create token classification model with 20 3Di labels
    class_config = ClassConfig(num_labels=20)
    model = T5EncoderForTokenClassification(base_model.config, class_config)
    
    # Transfer weights from base encoder
    model.shared = base_model.shared
    model.encoder = base_model.encoder
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
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
    
    # Move to device and set precision
    model = model.to(device)
    if device.type == "cpu":
        model.float()
    elif half_precision:
        model.half()
    
    return tokenizer, model


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
	Load tokenizer and token classification model for 3Di prediction.
	"""
	return load_token_classification_model(model_path, checkpoint_path)


def aa_to_3di_batch(tokenizer, model, seqs_raw, gen_kwargs=None):
	"""
	Convert a batch of AA sequences (strings) to 3Di strings using token classification.
	
	Args:
		tokenizer: T5 tokenizer
		model: T5EncoderForTokenClassification model
		seqs_raw: List of amino acid sequences
		gen_kwargs: Unused (kept for compatibility)
	
	Returns:
		List of 3Di sequences
	"""
	if len(seqs_raw) == 0:
		return []

	# Preprocess sequences
	inputs = [preprocess_sequence(s) for s in seqs_raw]
	
	# Store original sequence lengths (without preprocessing)
	seq_lengths = [len(s) for s in seqs_raw]

	# Tokenize
	ids = tokenizer.batch_encode_plus(
		inputs,
		return_tensors="pt",
		max_length=1024,
		padding=True,
		truncation=True,
		add_special_tokens=True
	).to(device)

	input_ids = ids["input_ids"]
	attention_mask = ids.get("attention_mask", None)

	# 3Di label mapping - 20 standard 3Di structural alphabet characters
	ss_mapping = {
		0: "A", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H", 7: "I",
		8: "K", 9: "L", 10: "M", 11: "N", 12: "P", 13: "Q", 14: "R",
		15: "S", 16: "T", 17: "V", 18: "W", 19: "Y"
	}

	with torch.no_grad():
		# Forward pass through token classification model
		outputs = model(
			input_ids=input_ids, 
			attention_mask=attention_mask
		)
		
		logits = outputs.logits  # Shape: (batch_size, seq_len, num_labels)
		
		# Get predicted class for each position (argmax)
		predictions = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)
		
		# Convert to numpy for processing
		predictions = predictions.cpu().numpy()
		attention_mask_np = attention_mask.cpu().numpy() if attention_mask is not None else None

	structure_sequences = []
	
	for batch_idx, (pred_seq, orig_len) in enumerate(zip(predictions, seq_lengths)):
		# Find valid token positions (ignore padding and special tokens)
		if attention_mask_np is not None:
			# Only consider positions where attention_mask is 1
			valid_positions = attention_mask_np[batch_idx] == 1
			valid_predictions = pred_seq[valid_positions]
		else:
			valid_predictions = pred_seq
		
		# Skip the first token (special token) and limit to original sequence length
		# The preprocessing adds "<AA2fold> " prefix, so we need to account for that
		# Typically first few tokens are special, and we want to extract predictions
		# corresponding to the actual amino acid positions
		
		# Find the start of actual sequence predictions
		# Skip special tokens at the beginning
		start_idx = 1  # Skip first special token
		
		# Extract predictions for actual sequence length
		sequence_predictions = valid_predictions[start_idx:start_idx + orig_len]
		
		# Convert predictions to 3Di characters
		structure_seq = "".join([
			ss_mapping.get(int(pred), "A")  # Default to "A" if prediction is out of range
			for pred in sequence_predictions
		])
		
		# Ensure we don't exceed original sequence length
		structure_seq = structure_seq[:orig_len]
		
		structure_sequences.append(structure_seq)

	return structure_sequences


def main():
	parser = argparse.ArgumentParser(
		description="Convert AA FASTA to 3Di token FASTA using ProstT5 token classification model."
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
		"--half-precision",
		action="store_true",
		help="Use half precision (float16) for faster inference on GPU.",
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
	tokenizer, model = load_token_classification_model(args.model_path, args.checkpoint, args.half_precision)

	# Token classification doesn't use generation parameters
	gen_kwargs_aa2fold = None

	# First pass: count total sequences for progress bar
	total_sequences = sum(1 for _ in read_fasta(args.input_fasta))
	
	batch_headers = []
	batch_seqs = []
	processed_sequences = 0

	with open(args.output_fasta, "w") as out_f:
		with tqdm(total=total_sequences, desc="Converting AA to 3Di", unit="seq") as pbar:
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
					
					processed_sequences += len(batch_seqs)
					pbar.update(len(batch_seqs))
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
				
				processed_sequences += len(batch_seqs)
				pbar.update(len(batch_seqs))


if __name__ == "__main__":
	# Examples:
	#
	# 1) Baseline ProstT5 token classification:
	#    python fasta23di.py input.fa output_3di.fa
	#
	# 2) With half precision for faster GPU inference:
	#    python fasta23di.py input.fa output_3di.fa --half-precision
	#
	# 3) Fine-tuned token classification model state_dict:
	#    torch.save(model.state_dict(), "ft_token_class_state_dict.pt")
	#    python fasta23di.py input.fa output_3di.fa \
	#        --model-path Rostlab/ProstT5 \
	#        --checkpoint ft_token_class_state_dict.pt
	#
	# 4) Fine-tuned full token classification model:
	#    torch.save(model, "ft_token_class_model.pt")
	#    python fasta23di.py input.fa output_3di.fa \
	#        --model-path Rostlab/ProstT5 \
	#        --checkpoint ft_token_class_model.pt --half-precision
	main()
