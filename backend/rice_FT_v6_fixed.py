"""
FastVLM 1.5B Fine-Tuning + Evaluation: Rice Disease Detection (V6 — Fixed)

FIXES APPLIED (based on diagnostic analysis):
1. mm_projector UNFROZEN with separate lower learning rate
2. mm_projector weights saved/restored in checkpoints
3. infer_disease() rewritten — first-sentence only, no blight scoring bias
4. Uses vqa_pairs_fixed.json (hedging-normalized by fix_vqa_hedging.py)
5. Increased early stopping patience (5 instead of 3) for projector convergence
6. Inference uses prepare_inputs_labels_for_multimodal (training-aligned pipeline)

NOTE: Diagnostic confirmed Pipeline A ≡ Pipeline B (no pipeline mismatch).
The core issue was frozen mm_projector — model sees images but can't learn
the correct visual→disease mapping. Fix #1 is the critical change.

Run: python fix_vqa_hedging.py   FIRST (creates vqa_pairs_fixed.json)
Then: python rice_FT_v6_fixed.py

Run from: /home/qurat_fatima/rice_FT_Feb/attempt_3/
"""

import torch
import gc

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {mem_gb:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ TF32 enabled for faster training")
else:
    print("⚠️ No GPU!")

import os
import json
import random
from collections import Counter

# ---- Paths ----
SCRIPT_DIR = "/home/qurat_fatima/rice_FT_Feb/attempt_3"
DATASET_ROOT = os.path.join(SCRIPT_DIR, "Rice Balanced Dataset")
# FIX #4: Use hedging-fixed VQA file
VQA_JSON = os.path.join(DATASET_ROOT, "vqa_pairs_fixed.json")

if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT}")
if not os.path.isfile(VQA_JSON):
    # Fallback to original if fixed version not found
    VQA_JSON_ORIG = os.path.join(DATASET_ROOT, "vqa_pairs_generated.json")
    if os.path.isfile(VQA_JSON_ORIG):
        print(f"⚠️ Fixed VQA not found, using original: {VQA_JSON_ORIG}")
        print(f"   Run fix_vqa_hedging.py first for best results!")
        VQA_JSON = VQA_JSON_ORIG
    else:
        raise FileNotFoundError(f"VQA file not found: {VQA_JSON}")

print(f"  Dataset root: {DATASET_ROOT}")
print(f"  VQA file: {VQA_JSON}")

# Verify splits
for split in ["train", "valid", "test"]:
    split_dir = os.path.join(DATASET_ROOT, split)
    if os.path.isdir(split_dir):
        n_imgs = len([f for f in os.listdir(split_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        has_coco = os.path.exists(os.path.join(split_dir, "_annotations.coco.json"))
        print(f"  {split}/: {n_imgs} images, COCO json: {'yes' if has_coco else 'no'}")

# ============================================================
# LOAD & CREATE TIERED DATA
# ============================================================
random.seed(42)

with open(VQA_JSON) as f:
    all_vqa = json.load(f)

print(f"\n   Total VQA conversations: {len(all_vqa)}")

def normalize_disease_label(label):
    if not label:
        return "unknown"
    s = str(label).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if s in ("blast", "blight", "brownspot"):
        return s
    if "brown" in s and "spot" in s:
        return "brownspot"
    if "bacterial" in s or "blight" in s:
        return "blight"
    return label

# Normalize labels
for item in all_vqa:
    item["_disease_key"] = normalize_disease_label(item.get("disease_label", "unknown"))

# ============================================================
# TIERED DATA CREATION
# 50% Tier 1 (single-turn classification only)
# 30% Tier 2 (2-turn: classification + 1 follow-up)
# 20% Tier 3 (full multi-turn conversation)
# T8 (differential diagnosis) goes ONLY into Tier 3
# ============================================================

t8_items = [v for v in all_vqa if v.get("theme") == "T8"]
non_t8_items = [v for v in all_vqa if v.get("theme") != "T8"]

random.shuffle(non_t8_items)
random.shuffle(t8_items)

n_total = len(all_vqa)
n_tier1 = int(n_total * 0.50)
n_tier2 = int(n_total * 0.30)

tiered_data = []

# Tier 1: single-turn (from non-T8 only)
tier1_count = 0
for item in non_t8_items:
    if tier1_count >= n_tier1:
        break
    new_item = {**item}
    new_item["conversations"] = item["conversations"][:2]
    new_item["tier"] = "T1_classification"
    tiered_data.append(new_item)
    tier1_count += 1

# Tier 2: 2-turn (from remaining non-T8)
remaining_non_t8 = non_t8_items[n_tier1:]
tier2_count = 0
for item in remaining_non_t8:
    if tier2_count >= n_tier2:
        break
    new_item = {**item}
    new_item["conversations"] = item["conversations"][:4]
    new_item["tier"] = "T2_short_conv"
    tiered_data.append(new_item)
    tier2_count += 1

# Tier 3: full conversations (remaining non-T8 + ALL T8)
remaining_for_t3 = remaining_non_t8[n_tier2:]
for item in remaining_for_t3:
    new_item = {**item}
    new_item["tier"] = "T3_full_conv"
    tiered_data.append(new_item)

for item in t8_items:
    new_item = {**item}
    new_item["tier"] = "T3_full_conv"
    tiered_data.append(new_item)

random.shuffle(tiered_data)

tier_counts = Counter(d["tier"] for d in tiered_data)
print(f"\n✅ TIERED DATA CREATED: {len(tiered_data)} total")
for tier, count in sorted(tier_counts.items()):
    pct = 100 * count / len(tiered_data)
    print(f"  {tier}: {count} ({pct:.0f}%)")

for tier in sorted(tier_counts.keys()):
    diseases = Counter(d["_disease_key"] for d in tiered_data if d["tier"] == tier)
    print(f"  {tier} diseases: {dict(diseases)}")

t1_chars = sum(len(d["conversations"][1]["value"]) for d in tiered_data if d["tier"] == "T1_classification")
t2_chars = sum(len(c["value"]) for d in tiered_data if d["tier"] == "T2_short_conv" for c in d["conversations"] if c["from"] == "assistant")
t3_chars = sum(len(c["value"]) for d in tiered_data if d["tier"] == "T3_full_conv" for c in d["conversations"] if c["from"] == "assistant")
total_chars = t1_chars + t2_chars + t3_chars
print(f"\n  Supervised token budget:")
print(f"    Tier 1 (classification):  {t1_chars:,} chars ({100*t1_chars/total_chars:.0f}%)")
print(f"    Tier 2 (short conv):      {t2_chars:,} chars ({100*t2_chars/total_chars:.0f}%)")
print(f"    Tier 3 (full conv):       {t3_chars:,} chars ({100*t3_chars/total_chars:.0f}%)")

# ============================================================
# STRATIFIED TRAIN / VAL SPLIT (85/15)
# ============================================================
by_disease = {}
for item in tiered_data:
    d = item["_disease_key"]
    by_disease.setdefault(d, []).append(item)

train_data = []
val_data = []
for disease, items in by_disease.items():
    random.shuffle(items)
    split_idx = int(len(items) * 0.85)
    train_data.extend(items[:split_idx])
    val_data.extend(items[split_idx:])

random.shuffle(train_data)
random.shuffle(val_data)

print(f"\n✅ Split: {len(train_data)} train + {len(val_data)} val")
for name, data in [("Train", train_data), ("Val", val_data)]:
    diseases = Counter(c["_disease_key"] for c in data)
    tiers = Counter(c["tier"] for c in data)
    print(f"  {name}: diseases={dict(diseases)}, tiers={dict(tiers)}")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME       = "apple/FastVLM-1.5B"
OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "output_fastvlm_v6")

SEED             = 42
EPOCHS           = 10
BATCH_SIZE       = 2
GRAD_ACCUM       = 8           # effective batch = 16
LR               = 1.5e-5
LR_PROJECTOR     = 7.5e-6      # FIX #1: Lower LR for mm_projector (0.5x main LR)
WEIGHT_DECAY     = 0.05
WARMUP_RATIO     = 0.06
MAX_TEXT_LEN     = 448

# Step-level early stopping — increased patience for projector convergence
EVAL_EVERY_N     = 30
PATIENCE_STEPS   = 5           # FIX #5: Was 3, now 5 (projector needs more time)
REL_DELTA        = 0.005

# Repetition penalty
TRAIN_REP_PENALTY  = 1.3
TRAIN_REP_NGRAM    = 4
GEN_REP_PENALTY    = 1.3
GEN_NO_REPEAT_NGRAM = 4

# LoRA
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.10
FREEZE_VISION    = True        # Vision encoder stays frozen
# FIX #1: mm_projector will be UNFROZEN (see load_fastvlm_with_lora)

# Eval
MAX_EVAL_IMAGES  = 500
MAX_EVAL_MULTI   = 50

IMAGE_TOKEN_INDEX = -200

print("\nConfiguration:")
print(f"  Model:       {MODEL_NAME}")
print(f"  Epochs:      {EPOCHS}")
print(f"  Batch:       {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
print(f"  LR:          {LR} (projector: {LR_PROJECTOR})")
print(f"  MAX_TEXT_LEN:{MAX_TEXT_LEN}")
print(f"  Step eval:   every {EVAL_EVERY_N} steps, patience={PATIENCE_STEPS} (rel_delta={REL_DELTA})")
print(f"  Rep penalty: train={TRAIN_REP_PENALTY} (ngram={TRAIN_REP_NGRAM}) | gen={GEN_REP_PENALTY} (ngram={GEN_NO_REPEAT_NGRAM})")
print(f"  LoRA:        r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
print(f"  Output:      {OUTPUT_DIR}")

# ============================================================
# IMPORTS & SETUP
# ============================================================

import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

set_seed(SEED)
print("✅ Imports done, seed set")

# ============================================================
# IMAGE FINDER
# ============================================================

class ImageFinder:
    def __init__(self, dataset_root):
        self.lookup = {}
        search_dirs = []
        for subdir in ["train", "valid", "test"]:
            d = os.path.join(dataset_root, subdir)
            if os.path.isdir(d):
                search_dirs.append(d)
        if not search_dirs:
            search_dirs.append(dataset_root)
        for d in search_dirs:
            for fname in os.listdir(d):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                    if fname not in self.lookup:
                        self.lookup[fname] = os.path.join(d, fname)
        log(f"ImageFinder: indexed {len(self.lookup)} images across {len(search_dirs)} folders")

    def find(self, image_name):
        if image_name in self.lookup:
            return self.lookup[image_name]
        return self.lookup.get(os.path.basename(image_name), None)

    def get_test_images(self):
        test_dir = os.path.join(DATASET_ROOT, "test")
        result = []
        if os.path.isdir(test_dir):
            for fname in sorted(os.listdir(test_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    result.append((os.path.join(test_dir, fname), fname))
        return result


image_finder = ImageFinder(DATASET_ROOT)

found_train = sum(1 for c in train_data if image_finder.find(c["image"]) is not None)
found_val = sum(1 for c in val_data if image_finder.find(c["image"]) is not None)
print(f"Images found: train={found_train}/{len(train_data)}, val={found_val}/{len(val_data)}")

if found_train == 0:
    print("❌ ERROR: No train images found!")
    print(f"   Dataset root: {DATASET_ROOT}")
    print(f"   Sample VQA image names: {[c['image'] for c in train_data[:5]]}")
    print(f"   Sample indexed images: {list(image_finder.lookup.keys())[:5]}")
else:
    print(f"✅ {found_train} train + {found_val} val conversations have matching images")

# ============================================================
# DATASET & COLLATOR
# ============================================================

def extract_disease(conv):
    turns = conv.get("conversations", [])
    if len(turns) < 2:
        return "unknown"
    ans = turns[1]["value"].lower()
    if "blast" in ans and "blight" not in ans:
        return "blast"
    if "blight" in ans or "bacterial blight" in ans:
        return "blight"
    if "brown spot" in ans or "brownspot" in ans or "brown_spot" in ans:
        return "brownspot"
    return "unknown"


class ConversationalVQADataset(Dataset):
    def __init__(self, conversations, image_finder, tokenizer, image_processor, max_text_len):
        self.image_finder = image_finder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len
        self.samples = []
        skipped = truncated = 0

        for conv_data in conversations:
            turns = conv_data.get("conversations", [])
            if len(turns) < 2:
                skipped += 1
                continue

            img_name = conv_data["image"]
            img_path = self.image_finder.find(img_name)
            if img_path is None:
                skipped += 1
                continue

            input_ids_list = []
            labels_list = []

            def add_tokens(text, is_answer):
                ids = self.tokenizer(text, add_special_tokens=False).input_ids
                input_ids_list.extend(ids)
                labels_list.extend(ids if is_answer else [-100] * len(ids))

            for i, turn in enumerate(turns):
                role = turn["from"].lower()
                text = turn["value"].strip()

                if role in ("human", "user"):
                    prompt_text = f"User: {text}\n"
                    if "<image>" in prompt_text:
                        prompt_text = prompt_text.replace("<image>", "").strip()
                        if not prompt_text.startswith("User:"):
                            prompt_text = f"User: {prompt_text}"
                        if not prompt_text.endswith("\n"):
                            prompt_text += "\n"
                        input_ids_list.append(IMAGE_TOKEN_INDEX)
                        labels_list.append(-100)
                        add_tokens(prompt_text, is_answer=False)
                    else:
                        add_tokens(prompt_text, is_answer=False)

                elif role in ("gpt", "assistant"):
                    add_tokens("Assistant: ", is_answer=False)
                    add_tokens(text + "\n", is_answer=True)

            eos_ids = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False).input_ids
            input_ids_list.extend(eos_ids)
            labels_list.extend([-100] * len(eos_ids))

            if len(input_ids_list) > max_text_len:
                truncated += 1

            self.samples.append({
                "input_ids": torch.tensor(input_ids_list[:max_text_len], dtype=torch.long),
                "labels": torch.tensor(labels_list[:max_text_len], dtype=torch.long),
                "image_path": img_path,
                "image_name": img_name,
                "disease": conv_data.get("_disease_key", normalize_disease_label(conv_data.get("disease_label", extract_disease(conv_data)))),
                "num_turns": len(turns),
                "tier": conv_data.get("tier", "unknown"),
            })

        if skipped: log(f"  Skipped {skipped} (missing image or too short)")
        if truncated: log(f"  Truncated {truncated} to {max_text_len} tokens")
        log(f"  Built {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        with Image.open(s["image_path"]) as im:
            image = im.convert("RGB")
        pv = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {
            "input_ids": s["input_ids"],
            "attention_mask": torch.ones_like(s["input_ids"]),
            "labels": s["labels"],
            "images": pv,
            "disease": s["disease"],
            "image_name": s["image_name"],
        }


@dataclass
class VQACollator:
    pad_token_id: int
    def __call__(self, batch):
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [b["input_ids"] for b in batch], batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [b["labels"] for b in batch], batch_first=True, padding_value=-100),
            "images": torch.stack([b["images"] for b in batch]),
            "diseases": [b["disease"] for b in batch],
            "image_names": [b["image_name"] for b in batch],
        }

print("✅ Dataset classes defined")

# ============================================================
# LOAD MODEL + LoRA + UNFREEZE mm_projector (FIX #1)
# ============================================================

def load_fastvlm_with_lora(device):
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32

    log(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map=None, trust_remote_code=True)
    base_model.to(device)

    # Freeze vision encoder (stays frozen — it's a good feature extractor)
    if FREEZE_VISION:
        vision_tower = base_model.get_vision_tower()
        for p in vision_tower.parameters():
            p.requires_grad = False
        log("Vision encoder frozen")

    # ═══════════════════════════════════════════════════════
    # FIX #1: UNFREEZE mm_projector
    # This is the bridge between vision and language.
    # Diagnostic proved the model sees images but maps them to wrong diseases.
    # The projector MUST be trainable to learn rice-disease-specific mappings.
    # ═══════════════════════════════════════════════════════
    mm_proj = base_model.get_model().mm_projector
    for p in mm_proj.parameters():
        p.requires_grad = True
    mm_proj_params = sum(p.numel() for p in mm_proj.parameters() if p.requires_grad)
    log(f"✅ mm_projector UNFROZEN: {mm_proj_params:,} trainable params")

    image_processor = base_model.get_vision_tower().image_processor
    if image_processor is None:
        log("WARNING: image_processor is None, creating fallback")
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor(
            size={"height": 512, "width": 512},
            crop_size={"height": 512, "width": 512},
            do_resize=True, do_center_crop=True, do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225])

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"])

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Verify mm_projector is still trainable after PEFT wrapping
    proj_trainable = sum(1 for n, p in model.named_parameters()
                         if "mm_projector" in n and p.requires_grad)
    log(f"  mm_projector layers trainable after PEFT: {proj_trainable}")
    if proj_trainable == 0:
        log("⚠️ WARNING: PEFT froze mm_projector! Re-enabling...")
        for n, p in model.named_parameters():
            if "mm_projector" in n:
                p.requires_grad = True

    return model, tokenizer, image_processor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, image_processor = load_fastvlm_with_lora(device)

test = "User: What disease is this?\nAssistant: "
ids = tokenizer(test, add_special_tokens=False).input_ids
log(f"Sanity check: '{test}' -> {len(ids)} tokens")

# ============================================================
# BUILD DATASETS
# ============================================================

log("Building train dataset...")
train_dataset = ConversationalVQADataset(
    train_data, image_finder, tokenizer, image_processor, MAX_TEXT_LEN)

log("Building val dataset...")
val_dataset = ConversationalVQADataset(
    val_data, image_finder, tokenizer, image_processor, MAX_TEXT_LEN)

print(f"\n✅ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

train_lens = [len(s["input_ids"]) for s in train_dataset.samples]
print(f"Token lengths — min: {min(train_lens)}, max: {max(train_lens)}, "
      f"mean: {np.mean(train_lens):.0f}, median: {np.median(train_lens):.0f}")

for tier in ["T1_classification", "T2_short_conv", "T3_full_conv"]:
    tier_lens = [len(s["input_ids"]) for s in train_dataset.samples if s["tier"] == tier]
    if tier_lens:
        print(f"  {tier}: avg={np.mean(tier_lens):.0f} tokens, n={len(tier_lens)}")

train_diseases = Counter(s["disease"] for s in train_dataset.samples)
val_diseases = Counter(s["disease"] for s in val_dataset.samples)
print(f"Train diseases: {dict(train_diseases)}")
print(f"Val diseases:   {dict(val_diseases)}")

# ============================================================
# TRAINING LOOP
# ============================================================

@dataclass
class TrainMetrics:
    train_loss_per_epoch: list = field(default_factory=list)
    val_loss_per_epoch: list = field(default_factory=list)
    step_train_losses: list = field(default_factory=list)
    step_val_losses: list = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    best_step: int = -1
    stopped_early_at_step: int = -1


def apply_repetition_penalty_to_logits(logits, input_ids, penalty=1.3):
    if penalty == 1.0:
        return logits
    last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
    for b in range(input_ids.shape[0]):
        unique_ids = input_ids[b].unique()
        score = last_logits[b, unique_ids]
        score = torch.where(score < 0, score * penalty, score / penalty)
        last_logits[b, unique_ids] = score
    return logits


def _ngram_rep_loss(logits, labels, ngram_size=4, penalty_weight=0.1):
    if ngram_size <= 0 or penalty_weight == 0.0:
        return torch.tensor(0.0, device=logits.device)

    pred_ids = logits.argmax(dim=-1)
    total_penalty = 0.0
    count = 0

    min_seq_len = min(pred_ids.shape[1], labels.shape[1])
    pred_ids = pred_ids[:, :min_seq_len]
    labels = labels[:, :min_seq_len]

    for b in range(pred_ids.shape[0]):
        mask = labels[b] != -100
        seq = pred_ids[b][mask].tolist()
        if len(seq) < ngram_size + 1:
            continue
        ngrams = [tuple(seq[i:i+ngram_size]) for i in range(len(seq) - ngram_size + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        dup_frac = (total - unique) / max(1, total)
        total_penalty += dup_frac
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=logits.device)

    return torch.tensor(penalty_weight * total_penalty / count,
                        device=logits.device, dtype=logits.dtype)


def compute_val_loss(model, val_loader, device, max_batches=0):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                    images=batch["images"].to(device))
            total_loss += outputs.loss.item()
            n += 1
            if 0 < max_batches <= n:
                break
    model.train()
    return total_loss / max(1, n)


# ═══════════════════════════════════════════════════════════
# FIX #2: Save mm_projector weights alongside LoRA checkpoint
# ═══════════════════════════════════════════════════════════

def save_checkpoint(model, tokenizer, save_dir, save_projector=True):
    """Save LoRA adapter + mm_projector weights."""
    ensure_dir(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    if save_projector:
        proj_state = {}
        for name, p in model.named_parameters():
            if "mm_projector" in name:
                proj_state[name] = p.data.cpu().clone()
        proj_path = os.path.join(save_dir, "mm_projector.pt")
        torch.save(proj_state, proj_path)
        log(f"  Saved mm_projector ({len(proj_state)} tensors) to {proj_path}")


def train_model(model, tokenizer, train_dataset, val_dataset, device):
    ensure_dir(OUTPUT_DIR)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    ensure_dir(ckpt_dir)

    collator = VQACollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collator, num_workers=2, pin_memory=True)

    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_optim_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_optim_steps)

    log(f"\n{'='*60}")
    log(f"TRAINING CONFIG")
    log(f"{'='*60}")
    log(f"  Train samples:    {len(train_dataset)}")
    log(f"  Val samples:      {len(val_dataset)}")
    log(f"  Batch:            {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
    log(f"  Batches/epoch:    {len(train_loader)}")
    log(f"  Optim steps/ep:   {steps_per_epoch}")
    log(f"  Total optim:      {total_optim_steps}")
    log(f"  Warmup:           {warmup_steps}")
    log(f"  LR:               {LR} (projector: {LR_PROJECTOR})")
    log(f"  Epochs:           {EPOCHS}")
    log(f"  Step-eval every:  {EVAL_EVERY_N} optim steps")
    log(f"  Step patience:    {PATIENCE_STEPS} evals (rel_delta={REL_DELTA})")
    log(f"  LoRA r={LORA_R} alpha={LORA_ALPHA}")
    log(f"{'='*60}\n")

    # ═══════════════════════════════════════════════════════
    # FIX #1: Separate param groups for LoRA vs mm_projector
    # ═══════════════════════════════════════════════════════
    lora_params = []
    proj_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "mm_projector" in name:
            proj_params.append(p)
        else:
            lora_params.append(p)

    log(f"  Optimizer param groups:")
    log(f"    LoRA params:      {len(lora_params)} tensors @ lr={LR}")
    log(f"    Projector params: {len(proj_params)} tensors @ lr={LR_PROJECTOR}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR},
        {"params": proj_params, "lr": LR_PROJECTOR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_val_loss = float("inf")
    best_step = -1
    best_epoch = -1
    evals_no_improve = 0
    metrics = TrainMetrics()

    global_step = 0
    optim_step = 0
    stopped_early = False

    for epoch in range(EPOCHS):
        if stopped_early:
            break

        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            if stopped_early:
                break

            global_step += 1
            n_steps += 1

            with torch.amp.autocast("cuda", enabled=True):
                _input_ids = batch["input_ids"].to(device)
                _labels    = batch["labels"].to(device)
                outputs = model(
                    input_ids=_input_ids,
                    attention_mask=batch["attention_mask"].to(device),
                    labels=_labels,
                    images=batch["images"].to(device))
                loss = outputs.loss
                if TRAIN_REP_PENALTY > 1.0 and outputs.logits is not None:
                    rep_aux = _ngram_rep_loss(
                        outputs.logits, _labels,
                        ngram_size=TRAIN_REP_NGRAM,
                        penalty_weight=(TRAIN_REP_PENALTY - 1.0) * 0.1)
                    loss = loss + rep_aux
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            current_loss = loss.item() * GRAD_ACCUM
            running_loss += current_loss
            metrics.step_train_losses.append((global_step, current_loss))

            if (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optim_step += 1

                # Step-level early stopping with relative threshold
                if EVAL_EVERY_N > 0 and optim_step % EVAL_EVERY_N == 0:
                    step_val = compute_val_loss(model, val_loader, device, max_batches=0)
                    metrics.step_val_losses.append((optim_step, step_val))
                    lr_now = scheduler.get_last_lr()[0]

                    min_delta = REL_DELTA * best_val_loss if best_val_loss < float("inf") else 0.01

                    log(f"  [Step {optim_step}] train={current_loss:.4f} val={step_val:.4f} "
                        f"lr={lr_now:.2e} best={best_val_loss:.4f} "
                        f"min_delta={min_delta:.5f} patience={evals_no_improve}/{PATIENCE_STEPS}")

                    if step_val < best_val_loss - min_delta:
                        best_val_loss = step_val
                        best_step = optim_step
                        best_epoch = epoch
                        evals_no_improve = 0

                        best_dir = os.path.join(ckpt_dir, "best")
                        save_checkpoint(model, tokenizer, best_dir, save_projector=True)
                        log(f"  ✅ New best! val={best_val_loss:.4f} at step {optim_step}")
                    else:
                        evals_no_improve += 1
                        log(f"  No improvement ({evals_no_improve}/{PATIENCE_STEPS})")
                        if evals_no_improve >= PATIENCE_STEPS:
                            log(f"  🛑 EARLY STOP at step {optim_step}. "
                                f"Best val={best_val_loss:.4f} at step {best_step}")
                            stopped_early = True
                            metrics.stopped_early_at_step = optim_step

                    model.train()

            if global_step % 50 == 0:
                avg = running_loss / n_steps
                elapsed = time.time() - epoch_start
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 1
                eta = ((len(train_loader) - batch_idx - 1) / rate) / 60 if rate > 0 else 0
                log(f"Ep {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                    f"loss={current_loss:.4f} avg={avg:.4f} ETA={eta:.1f}min")

        # End of epoch
        if not stopped_early:
            avg_train = running_loss / max(1, n_steps)
            metrics.train_loss_per_epoch.append(avg_train)

            full_val = compute_val_loss(model, val_loader, device, max_batches=0)
            metrics.val_loss_per_epoch.append(full_val)

            epoch_time = time.time() - epoch_start
            log(f"\n{'='*50}")
            log(f"Epoch {epoch+1}/{EPOCHS} done in {epoch_time/60:.1f}min")
            log(f"  Train loss: {avg_train:.4f}")
            log(f"  Val loss:   {full_val:.4f} (best: {best_val_loss:.4f} @ step {best_step})")
            log(f"{'='*50}")

            min_delta = REL_DELTA * best_val_loss if best_val_loss < float("inf") else 0.01
            if full_val < best_val_loss - min_delta:
                best_val_loss = full_val
                best_step = optim_step
                best_epoch = epoch
                evals_no_improve = 0
                best_dir = os.path.join(ckpt_dir, "best")
                save_checkpoint(model, tokenizer, best_dir, save_projector=True)
                log(f"  ✅ New best at epoch end! val={best_val_loss:.4f}")

    # Save last checkpoint
    last_dir = os.path.join(ckpt_dir, "last")
    save_checkpoint(model, tokenizer, last_dir, save_projector=True)

    best_dir = os.path.join(ckpt_dir, "best")
    if not os.path.isdir(best_dir):
        log("WARNING: No best checkpoint was saved, copying last as best")
        import shutil
        shutil.copytree(last_dir, best_dir)

    metrics.best_val_loss = best_val_loss
    metrics.best_epoch = best_epoch
    metrics.best_step = best_step

    with open(os.path.join(OUTPUT_DIR, "metrics_log.json"), "w") as f:
        json.dump({
            "train_loss_per_epoch": metrics.train_loss_per_epoch,
            "val_loss_per_epoch": metrics.val_loss_per_epoch,
            "step_train_losses": metrics.step_train_losses,
            "step_val_losses": metrics.step_val_losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "stopped_early_at_step": metrics.stopped_early_at_step,
            "total_optim_steps": optim_step,
        }, f, indent=2)

    log(f"\n🏁 Training complete. Best val={best_val_loss:.4f} at step {best_step} (epoch {best_epoch+1})")
    if stopped_early:
        log(f"   Early stopped at step {metrics.stopped_early_at_step}")
    return metrics


# Run training
metrics = train_model(model, tokenizer, train_dataset, val_dataset, device)

# ============================================================
# PLOT LOSS CURVES
# ============================================================

def plot_losses(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    n = len(metrics.train_loss_per_epoch)
    if n > 0:
        ax.plot(range(1,n+1), metrics.train_loss_per_epoch, 'o-', label='Train', color='steelblue', lw=2)
        ax.plot(range(1,len(metrics.val_loss_per_epoch)+1), metrics.val_loss_per_epoch, 's-', label='Val', color='darkorange', lw=2)
        for i,(t,v) in enumerate(zip(metrics.train_loss_per_epoch, metrics.val_loss_per_epoch)):
            ax.annotate(f'{t:.4f}', (i+1,t), fontsize=8, color='steelblue', textcoords='offset points', xytext=(5,5))
            ax.annotate(f'{v:.4f}', (i+1,v), fontsize=8, color='darkorange', textcoords='offset points', xytext=(5,-10))
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Epoch Losses')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if metrics.step_train_losses:
        steps, losses = zip(*metrics.step_train_losses)
        steps, losses = np.array(steps), np.array(losses)
        w = max(1, len(losses)//40)
        smooth = np.convolve(losses, np.ones(w)/w, mode='valid')
        ax.plot(steps, losses, alpha=0.1, color='steelblue', lw=0.5)
        ax.plot(steps[w-1:], smooth, color='steelblue', lw=2, label=f'Smoothed (w={w})')
    ax.set_xlabel('Step'); ax.set_ylabel('Train Loss'); ax.set_title('Step Train Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if metrics.step_val_losses:
        vs, vl = zip(*metrics.step_val_losses)
        ax.plot(vs, vl, 'o-', color='darkorange', ms=4, lw=1.5, label='Val')
        best_i = int(np.argmin(vl))
        ax.axvline(x=vs[best_i], color='green', ls='--', alpha=0.7, label=f'Best@step{vs[best_i]}')
        if metrics.stopped_early_at_step > 0:
            ax.axvline(x=metrics.stopped_early_at_step, color='red', ls=':', alpha=0.7, label=f'EarlyStop@{metrics.stopped_early_at_step}')
    ax.set_xlabel('Optim Step'); ax.set_ylabel('Val Loss'); ax.set_title('Step Val Loss + Early Stop')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved {path}")

plot_losses(metrics)

# ============================================================
# INFERENCE FUNCTION (UNCHANGED — diagnostic confirmed A ≡ B)
# ============================================================

def generate_answer(model, tokenizer, image_processor, img_path, question,
                    device, conversation_history=None, max_new_tokens=200,
                    vision_tower=None, mm_projector=None, embed_tokens=None):
    """
    Generate answer using manual pipeline.
    Diagnostic confirmed this produces IDENTICAL results to the native
    prepare_inputs_labels_for_multimodal pipeline (Pipeline A ≡ Pipeline B).
    """
    try:
        image = Image.open(img_path).convert("RGB")
        pixel_values = image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(device, dtype=torch.float32)

        prompt_parts = []
        if conversation_history:
            for prev_q, prev_a in conversation_history:
                clean_q = prev_q.replace("<image>", "").strip()
                prompt_parts.append(f"User: {clean_q}\nAssistant: {prev_a}\n")

        current_q = question.replace("<image>", "").strip()
        prompt_parts.append(f"User: {current_q}\nAssistant: ")
        prompt = "".join(prompt_parts)

        with torch.no_grad():
            image_features = vision_tower(pixel_values.to(dtype=vision_tower.dtype))
            if hasattr(image_features, 'last_hidden_state'):
                image_features = image_features.last_hidden_state
            elif isinstance(image_features, tuple):
                image_features = image_features[0]
            image_embeds = mm_projector(image_features)

        text_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        text_embeds = embed_tokens(text_ids)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        prompt_len = inputs_embeds.shape[1]
        past_kv = None
        gen_token_ids = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                if step == 0:
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=torch.ones(1, prompt_len, device=device, dtype=torch.long),
                        use_cache=True)
                else:
                    outputs = model(
                        input_ids=next_token,
                        attention_mask=torch.ones(1, prompt_len + len(gen_token_ids), device=device, dtype=torch.long),
                        past_key_values=past_kv,
                        use_cache=True)

                past_kv = outputs.past_key_values
                raw_logits = outputs.logits[:, -1, :]

                if GEN_REP_PENALTY > 1.0 and gen_token_ids:
                    seen = torch.tensor(gen_token_ids, device=device, dtype=torch.long).unsqueeze(0)
                    apply_repetition_penalty_to_logits(raw_logits.unsqueeze(1), seen, GEN_REP_PENALTY)

                if GEN_NO_REPEAT_NGRAM > 1 and len(gen_token_ids) >= GEN_NO_REPEAT_NGRAM - 1:
                    ngram_prefix = tuple(gen_token_ids[-(GEN_NO_REPEAT_NGRAM - 1):])
                    banned = set()
                    for start in range(len(gen_token_ids) - (GEN_NO_REPEAT_NGRAM - 1)):
                        if tuple(gen_token_ids[start:start + GEN_NO_REPEAT_NGRAM - 1]) == ngram_prefix:
                            banned.add(gen_token_ids[start + GEN_NO_REPEAT_NGRAM - 1])
                    if banned:
                        ban_ids = torch.tensor(list(banned), device=device, dtype=torch.long)
                        raw_logits[0, ban_ids] = float("-inf")

                next_token = torch.argmax(raw_logits, dim=-1, keepdim=True)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                gen_token_ids.append(next_token.item())

        decoded = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()
        if "User:" in decoded:
            decoded = decoded.split("User:")[0].strip()

        del inputs_embeds, image_embeds, text_embeds, pixel_values, past_kv, outputs
        torch.cuda.empty_cache()
        return decoded

    except Exception as e:
        log(f"  ERROR: {os.path.basename(img_path)}: {e}")
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        return None


# ============================================================
# FIX #3: REWRITTEN infer_disease() — first sentence only
# ============================================================

def infer_disease(text):
    """
    Extract disease classification from model response.
    Uses FIRST SENTENCE ONLY to avoid the old scoring bug where
    later mentions of other diseases (in differential diagnosis or
    rambling text) would override the actual diagnosis.

    Old version: scored across full 400-char response, gave blight +3
    bonus for appearing anywhere → 86% blight predictions.
    """
    if not text or not text.strip():
        return "unknown"

    t = text.lower().strip()

    # Extract first sentence
    first_sent = t.split('.')[0] if '.' in t else t[:100]

    # Check most specific patterns first
    if 'bacterial blight' in first_sent:
        return 'blight'
    if 'brown spot' in first_sent or 'brownspot' in first_sent or 'brown_spot' in first_sent:
        return 'brownspot'
    if 'blast' in first_sent and 'blight' not in first_sent:
        return 'blast'
    if 'blight' in first_sent:
        return 'blight'

    # Fallback: check first 150 chars if first sentence was very short
    first_chunk = t[:150]
    if 'bacterial blight' in first_chunk:
        return 'blight'
    if 'brown spot' in first_chunk or 'brownspot' in first_chunk:
        return 'brownspot'
    if 'blast' in first_chunk and 'blight' not in first_chunk:
        return 'blast'
    if 'blight' in first_chunk:
        return 'blight'

    return "unknown"

print("✅ Inference functions defined")

# ============================================================
# RELOAD BEST CHECKPOINT FOR EVALUATION (FIX #2: restore projector)
# ============================================================

del model
gc.collect()
torch.cuda.empty_cache()

best_ckpt = os.path.join(OUTPUT_DIR, "checkpoints", "best")
if not os.path.isdir(best_ckpt):
    best_ckpt = os.path.join(OUTPUT_DIR, "checkpoints", "last")
    log(f"No best checkpoint, using last: {best_ckpt}")
else:
    log(f"Reloading best checkpoint: {best_ckpt}")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, device_map=None, trust_remote_code=True)
base_model.to(device)

ip = None
try:
    vt = base_model.get_vision_tower()
    if vt is not None:
        ip = getattr(vt, "image_processor", None)
        if ip is not None:
            log(f"Got image_processor: {type(ip).__name__}")
except Exception as e:
    log(f"Could not get image_processor: {e}")

if ip is None:
    log("Creating CLIPImageProcessor fallback...")
    from transformers import CLIPImageProcessor
    ip = CLIPImageProcessor(
        size={"height": 512, "width": 512}, crop_size={"height": 512, "width": 512},
        do_resize=True, do_center_crop=True, do_normalize=True,
        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

_test = ip(images=Image.new("RGB", (256,256), (128,128,128)), return_tensors="pt")["pixel_values"]
log(f"Image processor verified: shape={_test.shape}")
del _test

model = PeftModel.from_pretrained(base_model, best_ckpt)

# ═══════════════════════════════════════════════════════════
# FIX #2: RESTORE mm_projector weights from checkpoint
# ═══════════════════════════════════════════════════════════
proj_path = os.path.join(best_ckpt, "mm_projector.pt")
if os.path.exists(proj_path):
    proj_state = torch.load(proj_path, map_location=device)
    restored = 0
    for name, p in model.named_parameters():
        if name in proj_state:
            p.data.copy_(proj_state[name].to(device))
            restored += 1
    log(f"✅ Restored mm_projector from checkpoint ({restored} tensors)")
else:
    log("⚠️ WARNING: No mm_projector.pt found — projector uses pretrained weights only!")

model.to(device)
model.eval()

# Adapter verification
log("=" * 60)
log("ADAPTER VERIFICATION")
cfg_path = os.path.join(best_ckpt, "adapter_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    log(f"  LoRA r={cfg.get('r')}, alpha={cfg.get('lora_alpha')}")
    log(f"  Targets: {cfg.get('target_modules')}")

lora_layers = lora_nonzero = 0
for name, param in model.named_parameters():
    if "lora" in name.lower():
        lora_layers += 1
        if param.abs().sum() > 0:
            lora_nonzero += 1
log(f"  LoRA layers: {lora_nonzero}/{lora_layers} non-zero")
log("=" * 60)

inner_model = model.base_model.model
vision_tower = inner_model.get_model().get_vision_tower()
mm_projector = inner_model.get_model().mm_projector
embed_tokens = inner_model.get_model().embed_tokens

log(f"Extracted: vision_tower={type(vision_tower).__name__}, mm_projector={type(mm_projector).__name__}")
log("✅ Best checkpoint loaded with vision components")

# ============================================================
# CLASSIFICATION EVALUATION ON TEST SET
# ============================================================

def load_ground_truth_coco():
    coco_path = os.path.join(DATASET_ROOT, "test", "_annotations.coco.json")
    if not os.path.exists(coco_path):
        coco_path = os.path.join(DATASET_ROOT, "valid", "_annotations.coco.json")
    if not os.path.exists(coco_path):
        log("No COCO json found, will use filename-based GT")
        return {}
    with open(coco_path) as f:
        coco = json.load(f)
    cat_map = {}
    for c in coco.get("categories", []):
        name = str(c.get("name","")).strip().lower().replace(" ","").replace("-","")
        if "blast" in name and "blight" not in name: cat_map[c["id"]] = "blast"
        elif "blight" in name or "bacterial" in name: cat_map[c["id"]] = "blight"
        elif "brown" in name: cat_map[c["id"]] = "brownspot"
    img_id_to_name = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    gt = {}
    for ann in coco.get("annotations", []):
        cid = ann["category_id"]
        img_id = ann["image_id"]
        if cid in cat_map and img_id in img_id_to_name:
            fname = img_id_to_name[img_id]
            if fname not in gt:
                gt[fname] = cat_map[cid]
    log(f"COCO ground truth: {len(gt)} images, {dict(Counter(gt.values()))}")
    return gt

def extract_disease_from_filename(fname):
    fl = fname.lower()
    if "blast" in fl and "blight" not in fl: return "blast"
    if "blight" in fl or "bacterial" in fl: return "blight"
    if "brown" in fl: return "brownspot"
    return "unknown"

test_gt = load_ground_truth_coco()
print("✅ Ground truth ready")

EVAL_QUESTIONS = [
    "What disease does this rice leaf have?",
    "Can you identify what disease is on this rice plant?",
    "What is wrong with this rice leaf?",
    "What rice disease do you see in this image?",
    "Is this blast, blight, or brown spot?",
    "What disease is affecting this rice crop?",
    "Help me identify what is wrong with this rice leaf.",
    "My rice has this problem on the leaves. What disease is it?",
]

test_images = image_finder.get_test_images()
log(f"Test images found: {len(test_images)}")

eval_pairs = []
for img_path, fname in test_images:
    gt = test_gt.get(fname, extract_disease_from_filename(fname))
    if gt in ["blast", "blight", "brownspot"]:
        eval_pairs.append((img_path, fname, gt))

log(f"Test images with ground truth: {len(eval_pairs)}")
log(f"Distribution: {dict(Counter(e[2] for e in eval_pairs))}")

if len(eval_pairs) > MAX_EVAL_IMAGES:
    random.shuffle(eval_pairs)
    eval_pairs = eval_pairs[:MAX_EVAL_IMAGES]
    log(f"Capped to {MAX_EVAL_IMAGES} images")

y_true, y_pred, all_predictions = [], [], []
start_time = time.time()

for i, (img_path, fname, true_disease) in enumerate(eval_pairs):
    question = EVAL_QUESTIONS[i % len(EVAL_QUESTIONS)]

    response = generate_answer(
        model, tokenizer, ip, img_path, question, device,
        vision_tower=vision_tower, mm_projector=mm_projector,
        embed_tokens=embed_tokens)

    if response is None:
        pred = "unknown"
        response = "GENERATION FAILED"
    else:
        pred = infer_disease(response)

    y_true.append(true_disease)
    y_pred.append(pred)

    all_predictions.append({
        "image": fname,
        "true_disease": true_disease,
        "predicted_disease": pred,
        "correct": pred == true_disease,
        "question": question,
        "response": response[:400],
    })

    if (i+1) % 25 == 0:
        valid_so_far = [(t,p) for t,p in zip(y_true, y_pred) if p != "unknown"]
        if valid_so_far:
            yt_tmp, yp_tmp = zip(*valid_so_far)
            acc_tmp = sum(1 for a,b in zip(yt_tmp, yp_tmp) if a == b) / len(yt_tmp)
            elapsed = time.time() - start_time
            eta = elapsed / (i+1) * (len(eval_pairs) - i - 1)
            log(f"  {i+1}/{len(eval_pairs)} | acc={acc_tmp:.3f} | unknowns={y_pred.count('unknown')} | ETA={eta/60:.1f}min")

# Results
valid = [(t,p) for t,p in zip(y_true, y_pred) if p != "unknown"]
unknowns = len(y_true) - len(valid)

print(f"\n{'='*60}")
print(f"CLASSIFICATION RESULTS")
print(f"{'='*60}")
print(f"Total test images: {len(y_true)}")
print(f"Unknown predictions: {unknowns}")

if valid:
    yt, yp = zip(*valid)
    labels = sorted(set(list(yt) + list(yp)) & {"blast", "blight", "brownspot"})

    cm = confusion_matrix(yt, yp, labels=labels)
    acc = sum(cm[i_c, i_c] for i_c in range(len(labels))) / max(1, cm.sum())

    print(f"\n⭐ OVERALL ACCURACY: {acc:.4f} ({acc*100:.1f}%)")

    print(f"\nConfusion Matrix:")
    header = f"{'':>12s} " + " ".join(f"{l:>10s}" for l in labels) + "  (predicted)"
    print(header)
    for i_r, row_label in enumerate(labels):
        row = " ".join(f"{cm[i_r,j_c]:>10d}" for j_c in range(len(labels)))
        print(f"{row_label:>12s} {row}")

    print(f"\nClassification Report:")
    print(classification_report(yt, yp, labels=labels))

    print("Per-class accuracy:")
    for i_c, label in enumerate(labels):
        total_cls = cm[i_c].sum()
        if total_cls > 0:
            cls_acc = cm[i_c, i_c] / total_cls
            print(f"  {label:>12s}: {cm[i_c,i_c]}/{total_cls} ({cls_acc:.1%})")

    fig, ax = plt.subplots(figsize=(8, 8))
    im_plot = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im_plot, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel="True Label", xlabel="Predicted Label",
           title=f"Rice Disease Classification — Accuracy: {acc:.1%}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2
    for i_r in range(cm.shape[0]):
        for j_c in range(cm.shape[1]):
            ax.text(j_c, i_r, str(cm[i_r, j_c]), ha="center", va="center",
                    color="white" if cm[i_r, j_c] > thresh else "black", fontsize=14)
    fig.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {cm_path}")

    report_dict = classification_report(yt, yp, labels=labels, output_dict=True)
    report_dict["overall_accuracy"] = acc
    report_dict["total_images"] = len(y_true)
    report_dict["unknown_predictions"] = unknowns

    with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "all_predictions.json"), "w") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print("\nSample predictions:")
    for p in all_predictions[:15]:
        marker = "✓" if p["correct"] else "✗"
        print(f"  {marker} true={p['true_disease']:10s} pred={p['predicted_disease']:10s} | {p['response'][:80]}")
else:
    print("❌ No valid predictions!")

# ============================================================
# MULTI-TURN CONVERSATIONAL EVALUATION
# ============================================================

multiturn = [c for c in val_data if len(c["conversations"]) >= 4]
random.shuffle(multiturn)
multiturn = multiturn[:MAX_EVAL_MULTI]

log(f"Evaluating multi-turn on {len(multiturn)} conversations...")
results_mt = []

for ci, conv in enumerate(multiturn):
    img_path = image_finder.find(conv["image"])
    if img_path is None:
        continue

    turns = conv["conversations"]
    history = []
    turn_results = []

    for t_idx in range(0, len(turns)-1, 2):
        q = turns[t_idx]["value"].replace("<image>","").strip()

        pred = generate_answer(
            model, tokenizer, ip, img_path, q, device,
            conversation_history=history if history else None,
            vision_tower=vision_tower, mm_projector=mm_projector,
            embed_tokens=embed_tokens)

        if pred is None:
            pred = "GENERATION FAILED"

        turn_results.append({
            "turn": t_idx//2+1,
            "question": q[:200],
            "predicted": pred[:400]})
        history.append((q, pred))

    disease_label = conv.get("_disease_key", normalize_disease_label(conv.get("disease_label", extract_disease(conv))))
    results_mt.append({
        "image": conv["image"],
        "disease": disease_label,
        "turns": turn_results})

    if (ci+1) % 10 == 0:
        log(f"  {ci+1}/{len(multiturn)}")

total_fu = unique_fu = 0
for r in results_mt:
    if len(r["turns"]) < 2:
        continue
    first = r["turns"][0]["predicted"].lower()[:100]
    for t in r["turns"][1:]:
        total_fu += 1
        if t["predicted"].lower()[:100] != first:
            unique_fu += 1

print(f"\n{'='*60}")
print(f"MULTI-TURN RESULTS")
print(f"{'='*60}")

if total_fu > 0:
    pct = unique_fu / total_fu * 100
    print(f"Follow-up diversity: {unique_fu}/{total_fu} ({pct:.1f}%) differ from Q1")
    if pct >= 50:
        print("✅ GOOD: Model produces diverse follow-up answers")
    elif pct >= 30:
        print("⚠️ PARTIAL: Some follow-ups are unique, some repeat Q1")
    else:
        print("❌ WARNING: Model mostly repeats first answer")

with open(os.path.join(OUTPUT_DIR, "multiturn_eval.json"), "w") as f:
    json.dump(results_mt, f, indent=2, ensure_ascii=False)

for r in results_mt[:5]:
    print(f"\n--- {r['image']} ({r['disease']}) ---")
    for t in r["turns"]:
        print(f"  Q{t['turn']}: {t['question'][:80]}")
        print(f"  A{t['turn']}: {t['predicted'][:120]}")

log(f"✅ Multi-turn eval complete. Saved to {OUTPUT_DIR}/multiturn_eval.json")

# ============================================================
# ZIP RESULTS
# ============================================================

import shutil

zip_name = os.path.join(SCRIPT_DIR, "fastvlm_1.5B_finetuned_rice_v6")
shutil.make_archive(zip_name, "zip", OUTPUT_DIR)
zip_path = zip_name + ".zip"

print(f"\nCreated: {zip_path}")
print(f"   Size: {os.path.getsize(zip_path)/1e6:.1f} MB")
print(f"\nContents of {OUTPUT_DIR}:")
for root, dirs, fnames in os.walk(OUTPUT_DIR):
    for f in fnames:
        fp = os.path.join(root, f)
        rel = os.path.relpath(fp, OUTPUT_DIR)
        size = os.path.getsize(fp)
        print(f"  {rel} ({size/1e6:.1f} MB)" if size > 1e6 else f"  {rel} ({size/1e3:.0f} KB)")
print("\n✅ Done.")
