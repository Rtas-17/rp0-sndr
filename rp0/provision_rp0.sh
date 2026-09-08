#!/bin/bash
# ============================================================
#  RP-0 PROVISIONING SCRIPT (vast.ai PROVISIONING_HOOK)
#  One-shot bootstrap on a fresh vast 1×3090 instance.
#  Goal: land the box EXACTLY on the locked RP-0 ship config
#  (SNDR dev748+TQ, MTP k=3, 65536 ctx, 3 users) with ZERO
#  manual intervention.
#
#  Laws baked in:
#   - swappiness 15 from the very start (no swap-tail growth)
#   - single engine only (port 18001; int4 rollback stays OFF)
#   - TQ_MAX == served ctx (65536) — THE configuration fix
#   - engine VmSwap target 0 kB, verified at end
#   - GPU orphan sweep before launch
# ============================================================

set -uo pipefail
LOG=/root/provision-rp0.log
exec > >(tee -a "$LOG") 2>&1

# ---- (A) memory hygiene immediately, before anything heavy ----
sysctl -w vm.swappiness=15 2>/dev/null || echo "sysctl write refused (container ro); vllm.sh wrapper will re-apply"
swapoff -a 2>/dev/null; swapon -a 2>/dev/null
echo "[prep] swappiness = $(cat /proc/sys/vm/swappiness), swap reset"

# ---- (B) work dirs ----
mkdir -p /root/models /root/qs2 /root/logs /root/rp0-bundle
bundle_url="${RP0_BUNDLE_URL:-https://raw.githubusercontent.com/REPLACE_ME/rp0/main/}"  # TODO: point at your private CDN / repo raw
if [ -n "${RP0_BUNDLE_ARCHIVE:-}" ]; then
    echo "[fx] RP0_BUNDLE_ARCHIVE provided as inline base64; using it"
else
    echo "[warn] no RP0_BUNDLE_ARCHIVE env; sourcing bundle files from /root/rp0-bundle if they exist"
fi

# ---- (C) install artifacts (from URL, or bundle placed at /root/rp0-bundle) ----
if [ -d /root/rp0-bundle ] && [ -s /root/rp0-bundle/sndr_start.sh ]; then
    echo "[ok] bundle present — no download needed"
else
    echo "[err] missing /root/rp0-bundle; PROVISIONING SCRIPT expects the bundle to be"
    echo "      baked into the docker image at /root/rp0-bundle (rp0-bigimage branch)"
    exit 1
fi
install -m 700 /root/rp0-bundle/sndr_start.sh /root/sndr_start.sh
install -m 640 /root/rp0-bundle/sndr-18001-env.sh /root/sndr-18001-env.sh
cp -n /root/rp0-bundle/qs18001.py /root/qs18001.py
cp -n /root/rp0-bundle/tq_test.py /root/rp0-bundle/tq_test_c2.py /root/rp0-bundle/tq_test_c4.py /root/ 2>/dev/null || true

# ---- (D) drop the model checkpoint in place (from image layers or hf) ----
MODEL_DIR=/root/models/Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "[fetch] downloading model from HF (noon-at-cgn source, ~15 GB)"
    /venv/main/bin/pip install -q "huggingface_hub[cli]" 2>/dev/null || true
    hf download noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound \
      --local-dir "$MODEL_DIR" 2>&1 | tail -2
fi
ls -la "$MODEL_DIR" | head -6

# ---- (E) venv-sndr: copy from image if baked, else build fresh ----
if [ ! -d /root/venv-sndr/lib/python3.12/site-packages/vllm ]; then
    echo "[build] venv-sndr absent — building per §2a (10-15 min)"
    python3 -m venv /root/venv-sndr
    source /root/venv-sndr/bin/activate
    pip install --upgrade pip setuptools wheel ninja setuptools-rust setuptools_scm >/dev/null
    pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
    git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git /root/sndr-vllm-src
    cd /root/sndr-vllm-src
    git fetch --depth 1 origin 2dfaae752b4db0d43cfc0715c780e33be030d0f1
    git checkout FETCH_HEAD
    VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation >/dev/null 2>&1
    sed -i "s/0.1.dev1+g2dfaae752/0.23.1rc1.dev748+g2dfaae752/" vllm/_version.py
    git clone https://github.com/Sandermage/sndr_core_engine.git /root/sndr
    pip install -e /root/sndr --no-deps >/dev/null 2>&1
else
    echo "[skip] /root/venv-sndr already baked into the image"
fi

# ---- (F) apply the two vllm source patches idempotently (see §2b) ----
source /root/venv-sndr/bin/activate
python3 <<'PYEOF'
import os, re
# Patch 1 — embed_tokens
qh = "/root/sndr-vllm-src/vllm/model_executor/models/qwen3_5.py"
s = open(qh).read()
if "quant_config=self.quant_config" not in s:
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
        open(qh,"w").write(s.replace(old,new,1)); print("patch-embed: applied")
    else:
        print("patch-embed: pattern drift, inspect manually")
PYEOF
# Patch 2 — draft head, script in image under /root/rp0-bundle/apply_patch2.py
if [ -f /root/rp0-bundle/apply_patch2.py ]; then
    python3 /root/rp0-bundle/apply_patch2.py
fi
cd /root/sndr-vllm-src && find vllm -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

# ---- (G) pin-gate verify ----
python3 -c "import sys; sys.path.insert(0,'/root/sndr'); \
  from sndr.engines.vllm.detection.guards import assert_vllm_pin_allowed; \
  r = assert_vllm_pin_allowed(); print('pin-gate:', r)"
[ "$?" = "0" ] || { echo "[FAIL] pin gate rejected — aborting boot."; exit 1; }

# ---- (H) port-sweep + wipe GPU orphans from the previous boot (paranoia) ----
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# ---- (I) LAUNCH the engine, port 18001, raw logging into /var/log/portal too ----
nohup bash -c 'tail -F -n 0 /root/sndr-18001-boot.log 2>/dev/null | while IFS= read -r l; do echo "[sndr-18001] $l"; done >> /var/log/portal/sndr.log' >/dev/null 2>&1 &
bash -c "nohup /root/sndr_start.sh > /root/sndr-18001-boot.log 2>&1 &"

# ---- (J) wait for ready, then verify health ----
echo "[wait] waiting up to 150 s for the engine API..."
for _ in $(seq 1 50); do
    H=$(curl -s -m 4 -o /dev/null -w "%{http_code}" --max-time 4 \
        -H "Authorization: Bearer genesis-local" http://127.0.0.1:18001/v1/models 2>/dev/null)
    [ "$H" = "200" ] && { echo "[OK] engine serving on 18001"; break; }
    sleep 3
done
if [ "$H" != "200" ]; then
    echo "[FAIL] engine did not come up. tail of boot log:"
    tail -20 /root/sndr-18001-boot.log
    exit 1
fi
grep -a "Maximum concurrency" /root/sndr-18001-boot.log | tail -1
for p in $(pgrep -f EngineCore); do
    s=$(grep VmSwap /proc/$p/status 2>/dev/null | awk '{print $2}')
    echo "pid $p VmSwap: $s kB"
done
echo "PROVISION RPO-0 COMPLETE — locked config live."

# ---- optional C1 smoke (usage-counted) ----
/venv/main/bin/python /root/qs18001.py 2>/dev/null || true
echo "=== PROVISION-COMPLETE ===" >> "$LOG"
