"""Rebuild the PaliGemma fast tokenizer from the public sentencepiece model (google/paligemma-3b-pt-224 is gated).

transformers 5.5 no longer auto-converts a .model file: GemmaTokenizerFast(vocab_file=...) silently builds a
5-token vocabulary. So the converter is driven directly, with a shim standing in for the slow tokenizer it
expects, and a <bos> prefix post-processor added the way PaliGemma encodes prompts.
"""
from pathlib import Path

import sentencepiece as spm
from tokenizers import processors
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import GemmaConverter, GemmaSentencePieceExtractor

_orig = GemmaSentencePieceExtractor.__init__


def _patched(self, model):          # transformers 5.5: __init__ parses the proto but never builds self.sp
    _orig(self, model)
    if not hasattr(self, "sp"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model)


GemmaSentencePieceExtractor.__init__ = _patched

SP = Path.home() / "pps/openpi_cache/big_vision/paligemma_tokenizer.model"
OUT = Path.home() / "yuna/ckpt/paligemma_tokenizer/hf"


class Shim:
    vocab_file = str(SP)
    pad_token, eos_token, bos_token, unk_token = "<pad>", "<eos>", "<bos>", "<unk>"
    add_prefix_space = False
    add_bos_token, add_eos_token = True, False


conv = GemmaConverter(Shim()).converted()
conv.post_processor = processors.TemplateProcessing(
    single="<bos> $A", pair="<bos> $A $B", special_tokens=[("<bos>", 2)])
tok = PreTrainedTokenizerFast(tokenizer_object=conv, bos_token="<bos>", eos_token="<eos>",
                              pad_token="<pad>", unk_token="<unk>", add_bos_token=True)
OUT.mkdir(parents=True, exist_ok=True)
tok.save_pretrained(str(OUT))
print("saved", OUT, "| vocab", tok.vocab_size)

sp = spm.SentencePieceProcessor(); sp.Load(str(SP))
bad = 0
for s in ["pick up the black bowl between the plate and the ramekin and place it on the plate",
          "Task: open the middle drawer of the cabinet, State: 128 64 255 0 17 200 3 9;\nAction: ",
          "close the top drawer of the cabinet and put the black bowl on top of it",
          "put both the alphabet soup and the tomato sauce in the basket"]:
    hf = tok(s, add_special_tokens=False)["input_ids"]
    ref = sp.EncodeAsIds(s)
    if hf != ref:
        bad += 1
        print("MISMATCH", repr(s[:45]))
        print("  hf ", hf[:14])
        print("  sp ", ref[:14])
print("bos prefix:", tok("hello")["input_ids"][:3], "| mismatches", bad, "/ 4")
