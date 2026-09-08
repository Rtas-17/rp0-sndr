#!/usr/bin/env python3
"""Idempotent installer for dev748 patch 2 (syv draft-head port) and patch 1
(quant-aware embed_tokens) on /root/sndr-vllm-src. Called from provision_rp0.sh.
Both patches are structural and safe to re-run: they check for their own signature
lines first and only patch if absent.

Run:  python3 /root/rp0-bundle/apply_vllm_patches.py
Then: cd /root/sndr-vllm-src && find vllm -name __pycache__ -exec rm -rf {} +
"""
import sys, re, os

VLLM_SRC = "/root/sndr-vllm-src"
ok = True

# ---------- Patch 1: quant-aware embed_tokens (qwen3_5.py) ----------
p1 = f"{VLLM_SRC}/vllm/model_executor/models/qwen3_5.py"
s = open(p1).read()
if "prefix=maybe_prefix(prefix, \"embed_tokens\")" in s and "quant_config=self.quant_config," in s.split("self.embed_tokens")[1][:400]:
    print("[patch1] already applied")
else:
    old = """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )"""
    new = """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )"""
    if old in s:
        open(p1, "w").write(s.replace(old, new, 1))
        print("[patch1] applied")
    else:
        print("[patch1] PATTERN DRIFT — manual inspection needed at line ~228")
        ok = False

# ---------- Patch 2: draft-head port (qwen3_5_mtp.py, four sites) ----------
p2 = f"{VLLM_SRC}/vllm/model_executor/models/qwen3_5_mtp.py"
s = open(p2).read()

# 2-A: ctor state on Qwen3_5MultiTokenPredictor
anchor_a = '        self.mtp_start_layer_idx = config.num_hidden_layers\n        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)'
add_a = anchor_a + """

        # syv patch: vocab-truncated draft head (idempotent installer)
        import os as _os
        self.draft_lm_head = None
        self.draft_vocab_ids = None
        _ids_path = _os.path.join(vllm_config.model_config.model, "mtp_draft_vocab_ids.pt")
        if _os.path.exists(_ids_path) and _os.environ.get("MTP_DRAFT_VOCAB", "1") != "0":
            import torch as _torch
            self.draft_vocab_ids = _torch.load(_ids_path, map_location="cpu")
            self.draft_lm_head = None  # constructed by Qwen3_5MTP below"""
if "syv patch: vocab-truncated draft head" in s:
    print("[patch2-A] already applied")
elif anchor_a in s:
    s = s.replace(anchor_a, add_a, 1); print("[patch2-A] applied")
else:
    print("[patch2-A] PATTERN DRIFT"); ok = False

# 2-B: draft head construction on Qwen3_5MTP
old_b = """        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids("""
new_b = """        self.logits_processor = LogitsProcessor(config.vocab_size)
        # syv patch: vocab-truncated draft head
        _ids = self.model.draft_vocab_ids
        if _ids is not None:
            self.draft_lm_head = ParallelLMHead(
                int(_ids.numel()),
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "mtp.draft_lm_head"),
            )
            self.draft_logits_processor = LogitsProcessor(int(_ids.numel()))
        else:
            self.draft_lm_head = None
            self.draft_logits_processor = None

    def embed_input_ids("""
if "self.draft_logits_processor = LogitsProcessor(int(_ids.numel()))" in s:
    print("[patch2-B] already applied")
elif old_b in s:
    s = s.replace(old_b, new_b, 1); print("[patch2-B] applied")
else:
    print("[patch2-B] PATTERN DRIFT"); ok = False

# 2-C: compute_logits draft branch
old_c = """        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:"""
new_c = """        if self.draft_logits_processor is not None:
            sub = self.draft_logits_processor(self.draft_lm_head, hidden_states)
            if sub is None:
                return None
            ids = self.model.draft_vocab_ids
            if ids.device != sub.device:
                ids = ids.to(sub.device)
                self.model.draft_vocab_ids = ids
            full = sub.new_full((sub.shape[0], self.config.vocab_size), float("-inf"))
            full.index_copy_(1, ids, sub)
            return full
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:"""
if "if self.draft_logits_processor is not None" in s:
    print("[patch2-C] already applied")
elif old_c in s:
    s = s.replace(old_c, new_c, 1); print("[patch2-C] applied")
else:
    print("[patch2-C] PATTERN DRIFT"); ok = False

# 2-D: load_weights draft-name pass-through
old_d = """            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")"""
new_d = """            for name, weight in weights:
                if "draft_lm_head" in name:
                    if self.draft_lm_head is None:
                        continue
                    yield name.replace("mtp.", "", 1), weight
                    continue
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")"""
if 'yield name.replace("mtp.", "", 1), weight' in s:
    print("[patch2-D] already applied")
elif old_d in s:
    s = s.replace(old_d, new_d, 1); print("[patch2-D] applied")
else:
    print("[patch2-D] PATTERN DRIFT"); ok = False

if not ok:
    print("[ABORT] one or more patch anchors drifted — inspect manually, do not boot")
    raise SystemExit(1)

open(p2, "w").write(s)
print("[OK] all patches applied. clear __pycache__ and relaunch.")
