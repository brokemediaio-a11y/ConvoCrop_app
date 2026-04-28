"""
Unseen Rice Disease Evaluation Script
Loads the finetuned ConvoCrop FastVLM-1.5B adapter and evaluates on real-life unseen images.

Three evaluation phases:
  Phase 1 -- Single-turn classification  (accuracy, F1, per-class metrics, confusion matrix)
  Phase 2 -- Multi-turn VQA with exact training-style questions (BLEU, ROUGE, METEOR, cosine sim)
  Phase 3 -- Multi-turn VQA with paraphrased questions (same NLP metrics)
"""

import os
import sys
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "apple/FastVLM-1.5B"
ADAPTER_PATH = r"C:\Saad\april_fastVLM1.5b\Convo_Crop\RiceFinetuning_V5\convocrop_best_adapter"
UNSEEN_DATASET_PATH = r"C:\Saad\april_fastVLM1.5b\Convo_Crop\RiceFinetuning_V5\real_life_dataset"

IMAGE_TOKEN_INDEX = -200
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.1

DISEASE_CLASSES = ["blast", "blight", "brownspot"]

DISEASE_KEYWORDS = {
    "blast": ["blast", "magnaporthe", "pyricularia", "spindle-shaped", "diamond-shaped"],
    "brownspot": ["brown spot", "brownspot", "brown_spot", "sesame-seed", "cochliobolus"],
    "blight": ["blight", "bacterial", "xanthomonas", "water-soaked"],
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 2. GROUND TRUTH (per-class reference VQA data for unseen images)
# ---------------------------------------------------------------------------

CLASSIFICATION_GROUND_TRUTH = {
    "blast": (
        "This leaf is infected with rice blast caused by Magnaporthe oryzae. "
        "Diamond or spindle-shaped lesions with grey-white centers and dark "
        "reddish-brown borders are visible on the leaf."
    ),
    "blight": (
        "This leaf shows bacterial leaf blight caused by Xanthomonas oryzae. "
        "Long water-soaked streaks running along the leaf with wavy, irregular "
        "margins and pale green to yellow discolouration."
    ),
    "brownspot": (
        "This leaf has brown spot disease caused by Cochliobolus miyabeanus. "
        "Circular to oval spots with light brown to grey centers and "
        "reddish-brown margins are scattered across the leaf surface."
    ),
}

EXACT_VQA_QUESTIONS = {
    "blast": [
        {
            "question": "What does the affected area look like?",
            "reference": (
                "Diamond or spindle-shaped lesions with grey-white centers and "
                "dark reddish-brown borders. Lesions shaped like small canoes or "
                "eyes with pale centers and dark edges."
            ),
        },
        {
            "question": "Can this infection spread to the rest of my field?",
            "reference": (
                "Wind-borne conidia that can travel 50-100 metres on a breeze. "
                "Rain splash moving spores downward and sideways to neighbouring "
                "plants. Infected seed and crop stubble serving as primary inoculum "
                "between seasons."
            ),
        },
        {
            "question": "What treatment should I apply?",
            "reference": (
                "Apply recommended fungicide at boot stage and around 80-90% "
                "heading if threshold is reached. Split nitrogen into 2-3 doses "
                "instead of one heavy application to keep leaves tougher. Use "
                "clean certified seed and avoid seed from blast-affected fields."
            ),
        },
        {
            "question": "What yield loss should I expect?",
            "reference": (
                "Rice blast typically causes 10-30% yield loss, with severe "
                "outbreaks reaching up to 80%. The economic impact depends on "
                "whether neck blast develops at heading."
            ),
        },
        {
            "question": "What role does soil nutrition play?",
            "reference": (
                "Balance nitrogen with potassium since potassium strengthens cell "
                "walls against fungal penetration. Excess nitrogen softens leaf "
                "tissue and makes blast worse. Split nitrogen into 2-3 smaller doses."
            ),
        },
    ],
    "blight": [
        {
            "question": "How do the marks in this image compare to a textbook case?",
            "reference": (
                "Long water-soaked streaks running along the leaf with wavy, "
                "irregular margins. Pale green to yellow streaks with translucent "
                "halos at the advancing front."
            ),
        },
        {
            "question": "Can this infection spread to the rest of my field?",
            "reference": (
                "Bacterial ooze entering through leaf hydathodes and wounds. "
                "Rain splash and wind spreading bacteria from plant to plant. "
                "This spreads faster in flooded lowland conditions where bacteria "
                "move freely in water."
            ),
        },
        {
            "question": "What treatment should I apply?",
            "reference": (
                "No curative chemical spray is available for bacterial blight. "
                "Reduce nitrogen application since excess nitrogen worsens blight. "
                "Drain excess water from fields to slow bacterial movement. "
                "Remove severely infected plants to reduce inoculum."
            ),
        },
        {
            "question": "What yield loss should I expect?",
            "reference": (
                "Bacterial leaf blight can cause yield losses of 20-30% in "
                "moderate cases, and up to 50-80% in severe epidemics. "
                "Blight is more severe on basmati varieties than on coarse or "
                "IRRI varieties in Punjab."
            ),
        },
        {
            "question": "Does nitrogen management matter here?",
            "reference": (
                "Excess nitrogen makes blight worse by softening leaf tissue and "
                "promoting lush growth that bacteria love. Split nitrogen into 2-3 "
                "doses and avoid heavy top-dressing. Balance with potassium to "
                "strengthen plant defences."
            ),
        },
    ],
    "brownspot": [
        {
            "question": "What does the affected area look like?",
            "reference": (
                "Circular to oval spots with light brown to grey centers and "
                "reddish-brown margins. Small discrete brown dots scattered "
                "across the leaf surface, resembling sesame seeds."
            ),
        },
        {
            "question": "Can this infection spread to the rest of my field?",
            "reference": (
                "Brown spot spreads through wind-borne conidia and infected seed. "
                "The fungus survives on crop debris and seed between seasons. "
                "Warm humid conditions with temperatures around 25-30 degrees "
                "celsius favour disease development."
            ),
        },
        {
            "question": "What treatment should I apply?",
            "reference": (
                "Brown spot is fundamentally a soil nutrition problem. Low potassium "
                "and silicon are the main gaps feeding this disease. Get a soil test "
                "and address deficiencies. Fixing fertility reduces brown spot more "
                "than spraying alone."
            ),
        },
        {
            "question": "What yield loss should I expect?",
            "reference": (
                "Brown spot typically causes 10-20% yield loss. In severe cases "
                "with poor soil nutrition, losses can reach up to 40-50%. "
                "Grain quality is also affected with discoloured kernels."
            ),
        },
        {
            "question": "Should I change my fertilizer approach because of this?",
            "reference": (
                "Brown spot thrives on potassium-deficient or silicon-deficient "
                "soils. Get a soil test and apply potash if K is low. Adding "
                "silicon sources like rice husk ash can also help. Balanced "
                "NPK application is key to controlling this disease."
            ),
        },
    ],
}

PARAPHRASED_VQA_QUESTIONS = {
    "blast": [
        {
            "question": "Describe the visual characteristics of the damage on this leaf.",
            "reference": EXACT_VQA_QUESTIONS["blast"][0]["reference"],
        },
        {
            "question": "Is there a risk of this disease moving to nearby plants?",
            "reference": EXACT_VQA_QUESTIONS["blast"][1]["reference"],
        },
        {
            "question": "What management steps should I take for this condition?",
            "reference": EXACT_VQA_QUESTIONS["blast"][2]["reference"],
        },
        {
            "question": "How much harvest reduction could this cause?",
            "reference": EXACT_VQA_QUESTIONS["blast"][3]["reference"],
        },
        {
            "question": "How should I adjust my fertilizer plan in response to this?",
            "reference": EXACT_VQA_QUESTIONS["blast"][4]["reference"],
        },
    ],
    "blight": [
        {
            "question": "Walk me through what the symptoms on this leaf look like.",
            "reference": EXACT_VQA_QUESTIONS["blight"][0]["reference"],
        },
        {
            "question": "Could neighbouring rows get infected from this?",
            "reference": EXACT_VQA_QUESTIONS["blight"][1]["reference"],
        },
        {
            "question": "What is the best course of action for managing this disease?",
            "reference": EXACT_VQA_QUESTIONS["blight"][2]["reference"],
        },
        {
            "question": "How badly will this hurt my overall crop production?",
            "reference": EXACT_VQA_QUESTIONS["blight"][3]["reference"],
        },
        {
            "question": "Does the amount of fertilizer I use affect this infection?",
            "reference": EXACT_VQA_QUESTIONS["blight"][4]["reference"],
        },
    ],
    "brownspot": [
        {
            "question": "Can you tell me what the spots on the leaf look like in detail?",
            "reference": EXACT_VQA_QUESTIONS["brownspot"][0]["reference"],
        },
        {
            "question": "How easily does this disease travel between plants?",
            "reference": EXACT_VQA_QUESTIONS["brownspot"][1]["reference"],
        },
        {
            "question": "What practical steps can I take right now to deal with this?",
            "reference": EXACT_VQA_QUESTIONS["brownspot"][2]["reference"],
        },
        {
            "question": "What is the financial risk if I do nothing about this?",
            "reference": EXACT_VQA_QUESTIONS["brownspot"][3]["reference"],
        },
        {
            "question": "Is there a connection between my soil nutrients and this disease?",
            "reference": EXACT_VQA_QUESTIONS["brownspot"][4]["reference"],
        },
    ],
}


# ---------------------------------------------------------------------------
# 3. UNSEEN DATASET LOADING
# ---------------------------------------------------------------------------

def load_unseen_dataset(dataset_path):
    """Scan real_life_dataset subfolders and build image list with ground truth."""
    samples = []
    for disease_folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, disease_folder)
        if not os.path.isdir(folder_path):
            continue
        disease_label = disease_folder.lower()
        if disease_label not in DISEASE_CLASSES:
            print(f"  Skipping unknown folder: {disease_folder}")
            continue
        for img_file in sorted(os.listdir(folder_path)):
            ext = img_file.lower().rsplit(".", 1)[-1] if "." in img_file else ""
            if ext not in ("jpg", "jpeg", "png", "webp"):
                continue
            samples.append({
                "image_path": os.path.join(folder_path, img_file),
                "image_name": img_file,
                "disease": disease_label,
                "folder": disease_folder,
            })
    return samples


# ---------------------------------------------------------------------------
# 4. MODEL LOADING
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(base_model_id, adapter_path, device="cuda"):
    """Load FastVLM base model + LoRA adapter."""
    print(f"Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    except Exception as e:
        print(f"  Adapter tokenizer failed ({e}), falling back to base model tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 5. INFERENCE (fixed version from the notebook)
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, image_path, conversation_history, device="cuda"):
    """
    Run inference with the fixed decoding strategy.
    Decodes the full output and extracts the response via string matching
    on the last <|im_start|>assistant marker.
    """
    model.eval()
    base_model = model.base_model if hasattr(model, "base_model") else model

    image = Image.open(image_path).convert("RGB")
    vt = (
        base_model.get_vision_tower()
        if hasattr(base_model, "get_vision_tower")
        else base_model.model.get_vision_tower()
    )
    pixel_values = vt.image_processor(images=image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device, dtype=torch.float16)

    messages = []
    for turn in conversation_history:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})

    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    if "<image>" in rendered:
        pre, post = rendered.split("<image>", 1)
        pre_ids = tokenizer(pre, add_special_tokens=False, return_tensors="pt").input_ids
        post_ids = tokenizer(post, add_special_tokens=False, return_tensors="pt").input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(device)
    else:
        input_ids = tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output_ids = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=pixel_values,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    full_decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    assistant_marker = "<|im_start|>assistant\n"
    last_marker_pos = full_decoded.rfind(assistant_marker)

    if last_marker_pos != -1:
        response = full_decoded[last_marker_pos + len(assistant_marker):]
        for end_tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
            response = response.split(end_tok)[0]
        response = response.strip()
    else:
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_text = rendered.replace("<image>", "").strip()
        if prompt_text in response:
            response = response[response.index(prompt_text) + len(prompt_text):].strip()

    return response


# ---------------------------------------------------------------------------
# 6. CLASSIFICATION EXTRACTION
# ---------------------------------------------------------------------------

def extract_disease_from_response(response_text):
    """Extract the predicted disease class from model response using keyword matching."""
    response_lower = response_text.lower()
    pred_disease = "unknown"
    max_score = 0
    for disease, keywords in DISEASE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in response_lower)
        if score > max_score:
            max_score = score
            pred_disease = disease
    return pred_disease


# ---------------------------------------------------------------------------
# 7. NLP METRICS
# ---------------------------------------------------------------------------

def compute_nlp_metrics(predictions, references):
    """
    Compute BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR, and Cosine Similarity
    between predicted responses and reference answers.
    Returns per-sample and aggregate metrics.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer

    import nltk
    for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    per_sample = []
    all_bleu, all_r1, all_r2, all_rl, all_meteor, all_cos = [], [], [], [], [], []

    vectorizer = TfidfVectorizer()
    combined = predictions + references
    if len(combined) < 2 or all(len(t.strip()) == 0 for t in combined):
        cos_scores = [0.0] * len(predictions)
    else:
        try:
            tfidf_matrix = vectorizer.fit_transform(combined)
            n = len(predictions)
            cos_scores = []
            for i in range(n):
                sim = sklearn_cosine(tfidf_matrix[i:i+1], tfidf_matrix[n+i:n+i+1])[0][0]
                cos_scores.append(float(sim))
        except Exception:
            cos_scores = [0.0] * len(predictions)

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())

        if len(pred_tokens) == 0:
            bleu = 0.0
        else:
            bleu = sentence_bleu(
                [ref_tokens], pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoother,
            )

        rouge_scores = scorer.score(ref, pred)
        r1 = rouge_scores["rouge1"].fmeasure
        r2 = rouge_scores["rouge2"].fmeasure
        rl = rouge_scores["rougeL"].fmeasure

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            meteor = 0.0
        else:
            meteor = nltk_meteor([ref_tokens], pred_tokens)

        cos = cos_scores[i]

        sample_metrics = {
            "bleu": bleu,
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rl,
            "meteor": meteor,
            "cosine_similarity": cos,
        }
        per_sample.append(sample_metrics)
        all_bleu.append(bleu)
        all_r1.append(r1)
        all_r2.append(r2)
        all_rl.append(rl)
        all_meteor.append(meteor)
        all_cos.append(cos)

    aggregate = {
        "bleu": float(np.mean(all_bleu)) if all_bleu else 0.0,
        "rouge1": float(np.mean(all_r1)) if all_r1 else 0.0,
        "rouge2": float(np.mean(all_r2)) if all_r2 else 0.0,
        "rougeL": float(np.mean(all_rl)) if all_rl else 0.0,
        "meteor": float(np.mean(all_meteor)) if all_meteor else 0.0,
        "cosine_similarity": float(np.mean(all_cos)) if all_cos else 0.0,
    }
    return aggregate, per_sample


# ---------------------------------------------------------------------------
# 8. PHASE 1 -- Single-turn Classification
# ---------------------------------------------------------------------------

def run_classification(model, tokenizer, samples, device="cuda"):
    """Run single-turn classification on every unseen image."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Single-Turn Classification")
    print("=" * 70)

    classification_questions = [
        "<image>\nWhat disease is on this rice leaf?",
        "<image>\nIdentify the disease on this leaf.",
        "<image>\nIs this rice leaf healthy or diseased? If diseased, name the disease.",
        "<image>\nExamine this image and identify any disease present.",
        "<image>\nI took this photo in my rice field. What disease is this?",
    ]

    results = []
    preds = []
    gts = []

    for sample in tqdm(samples, desc="Phase 1 - Classification"):
        question = random.choice(classification_questions)
        conversation = [{"from": "human", "value": question}]

        try:
            response = generate_response(model, tokenizer, sample["image_path"], conversation, device)
        except Exception as e:
            print(f"  Error on {sample['image_name']}: {e}")
            response = ""

        pred_disease = extract_disease_from_response(response)
        true_disease = sample["disease"]

        preds.append(pred_disease)
        gts.append(true_disease)
        results.append({
            "image": sample["image_name"],
            "folder": sample["folder"],
            "true_disease": true_disease,
            "predicted_disease": pred_disease,
            "correct": pred_disease == true_disease,
            "question": question,
            "response": response[:500],
        })

    return results, preds, gts


def compute_classification_metrics(preds, gts, class_labels):
    """Compute comprehensive classification metrics."""
    metrics = {}

    metrics["overall_accuracy"] = accuracy_score(gts, preds)
    metrics["macro_f1"] = f1_score(gts, preds, labels=class_labels, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(gts, preds, labels=class_labels, average="weighted", zero_division=0)
    metrics["macro_precision"] = precision_score(gts, preds, labels=class_labels, average="macro", zero_division=0)
    metrics["weighted_precision"] = precision_score(gts, preds, labels=class_labels, average="weighted", zero_division=0)
    metrics["macro_recall"] = recall_score(gts, preds, labels=class_labels, average="macro", zero_division=0)
    metrics["weighted_recall"] = recall_score(gts, preds, labels=class_labels, average="weighted", zero_division=0)

    per_class_acc = {}
    for cls in class_labels:
        cls_indices = [i for i, g in enumerate(gts) if g == cls]
        if len(cls_indices) > 0:
            correct = sum(1 for i in cls_indices if preds[i] == gts[i])
            per_class_acc[cls] = correct / len(cls_indices)
        else:
            per_class_acc[cls] = 0.0
    metrics["per_class_accuracy"] = per_class_acc

    report_dict = classification_report(
        gts, preds, labels=class_labels, target_names=class_labels,
        output_dict=True, zero_division=0
    )
    metrics["classification_report"] = report_dict

    cm = confusion_matrix(gts, preds, labels=class_labels)
    metrics["confusion_matrix"] = cm.tolist()

    unknown_count = sum(1 for p in preds if p == "unknown")
    metrics["unknown_predictions"] = unknown_count

    return metrics


# ---------------------------------------------------------------------------
# 9. PHASE 2 -- Multi-turn VQA with Exact Questions
# ---------------------------------------------------------------------------

def run_multiturn_vqa(model, tokenizer, samples, question_bank, phase_name, device="cuda"):
    """
    Run multi-turn VQA on all unseen images.
    For each image: send classification question first, then follow-up questions
    from the question_bank, building a running conversation history.
    """
    print(f"\n{'=' * 70}")
    print(f"  {phase_name}")
    print("=" * 70)

    all_predictions = []
    all_references = []
    detailed_results = []

    for sample in tqdm(samples, desc=phase_name):
        disease = sample["disease"]
        questions = question_bank.get(disease, [])
        if not questions:
            continue

        classification_q = "<image>\nWhat disease is on this rice leaf?"
        conversation_history = [{"from": "human", "value": classification_q}]

        try:
            class_response = generate_response(
                model, tokenizer, sample["image_path"], conversation_history, device
            )
        except Exception as e:
            print(f"  Error on {sample['image_name']} (classification): {e}")
            class_response = ""

        conversation_history.append({"from": "gpt", "value": class_response})

        image_results = {
            "image": sample["image_name"],
            "disease": disease,
            "classification_response": class_response[:300],
            "turns": [],
        }

        for qa in questions:
            conversation_history.append({"from": "human", "value": qa["question"]})

            try:
                response = generate_response(
                    model, tokenizer, sample["image_path"], conversation_history, device
                )
            except Exception as e:
                print(f"  Error on {sample['image_name']} ({qa['question'][:30]}): {e}")
                response = ""

            conversation_history.append({"from": "gpt", "value": response})

            all_predictions.append(response)
            all_references.append(qa["reference"])

            image_results["turns"].append({
                "question": qa["question"],
                "predicted": response[:500],
                "reference": qa["reference"][:500],
            })

        detailed_results.append(image_results)

    return all_predictions, all_references, detailed_results


# ---------------------------------------------------------------------------
# 10. VISUALIZATION
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, labels, title, save_path):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_class_accuracy(per_class_acc, title, save_path):
    """Plot per-class accuracy bar chart."""
    classes = list(per_class_acc.keys())
    accs = [per_class_acc[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_nlp_metrics_comparison(exact_agg, paraphrase_agg, save_path):
    """Plot side-by-side comparison of NLP metrics for exact vs paraphrased."""
    metric_names = ["bleu", "rouge1", "rouge2", "rougeL", "meteor", "cosine_similarity"]
    display_names = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "Cosine Sim"]

    exact_vals = [exact_agg.get(m, 0) for m in metric_names]
    para_vals = [paraphrase_agg.get(m, 0) for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, exact_vals, width, label="Exact Questions", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, para_vals, width, label="Paraphrased Questions", color="#DD8452")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("NLP Metrics: Exact vs Paraphrased VQA Questions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(exact_vals), max(para_vals)) * 1.25 + 0.05)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_class_nlp_metrics(detailed_results, question_bank, title_suffix, save_path):
    """Plot per-class breakdown of NLP metrics."""
    class_preds = defaultdict(list)
    class_refs = defaultdict(list)

    for result in detailed_results:
        disease = result["disease"]
        for turn in result["turns"]:
            class_preds[disease].append(turn["predicted"])
            class_refs[disease].append(turn["reference"])

    metric_names = ["bleu", "rouge1", "rougeL", "meteor", "cosine_similarity"]
    display_names = ["BLEU", "ROUGE-1", "ROUGE-L", "METEOR", "Cosine Sim"]
    class_labels = sorted(class_preds.keys())

    data = {m: [] for m in metric_names}
    for cls in class_labels:
        if class_preds[cls]:
            agg, _ = compute_nlp_metrics(class_preds[cls], class_refs[cls])
        else:
            agg = {m: 0.0 for m in metric_names}
        for m in metric_names:
            data[m].append(agg[m])

    x = np.arange(len(class_labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (m, dn) in enumerate(zip(metric_names, display_names)):
        offset = (idx - len(metric_names) / 2 + 0.5) * width
        ax.bar(x + offset, data[m], width, label=dn)

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Class NLP Metrics -- {title_suffix}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 11. REPORT PRINTING
# ---------------------------------------------------------------------------

def print_classification_report(metrics):
    """Print Phase-1 classification results to console."""
    print("\n" + "=" * 70)
    print("  CLASSIFICATION RESULTS (Phase 1)")
    print("=" * 70)
    print(f"  Overall Accuracy:      {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']:.1%})")
    print(f"  Macro F1-Score:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score:     {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision:       {metrics['macro_precision']:.4f}")
    print(f"  Weighted Precision:    {metrics['weighted_precision']:.4f}")
    print(f"  Macro Recall:          {metrics['macro_recall']:.4f}")
    print(f"  Weighted Recall:       {metrics['weighted_recall']:.4f}")
    print(f"  Unknown Predictions:   {metrics['unknown_predictions']}")

    print("\n  Per-Class Accuracy:")
    for cls, acc in metrics["per_class_accuracy"].items():
        n = sum(1 for g in [cls] if True)
        print(f"    {cls:15s}: {acc:.4f} ({acc:.1%})")

    report = metrics["classification_report"]
    print("\n  Detailed Classification Report:")
    header = f"  {'':15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}"
    print(header)
    for cls in DISEASE_CLASSES:
        if cls in report:
            r = report[cls]
            print(f"  {cls:15s} {r['precision']:10.4f} {r['recall']:10.4f} {r['f1-score']:10.4f} {int(r['support']):10d}")


def print_nlp_report(aggregate, phase_name):
    """Print NLP metrics summary."""
    print(f"\n{'=' * 70}")
    print(f"  NLP METRICS -- {phase_name}")
    print("=" * 70)
    print(f"  BLEU:              {aggregate['bleu']:.4f}")
    print(f"  ROUGE-1:           {aggregate['rouge1']:.4f}")
    print(f"  ROUGE-2:           {aggregate['rouge2']:.4f}")
    print(f"  ROUGE-L:           {aggregate['rougeL']:.4f}")
    print(f"  METEOR:            {aggregate['meteor']:.4f}")
    print(f"  Cosine Similarity: {aggregate['cosine_similarity']:.4f}")


# ---------------------------------------------------------------------------
# 12. MAIN
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "evaluation_results",
        f"unseen_rice_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -- Load dataset --
    print("\nLoading unseen dataset...")
    samples = load_unseen_dataset(UNSEEN_DATASET_PATH)
    print(f"  Found {len(samples)} images across {len(DISEASE_CLASSES)} disease classes")
    for cls in DISEASE_CLASSES:
        n = sum(1 for s in samples if s["disease"] == cls)
        print(f"    {cls}: {n} images")

    # -- Load model --
    print()
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_ID, ADAPTER_PATH, device)

    # ===================================================================
    # PHASE 1: Single-turn classification
    # ===================================================================
    class_results, preds, gts = run_classification(model, tokenizer, samples, device)
    class_metrics = compute_classification_metrics(preds, gts, DISEASE_CLASSES)
    print_classification_report(class_metrics)

    cm = np.array(class_metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm, DISEASE_CLASSES,
        "Confusion Matrix -- Unseen Rice Dataset",
        os.path.join(output_dir, "confusion_matrix.png"),
    )
    plot_per_class_accuracy(
        class_metrics["per_class_accuracy"],
        "Per-Class Accuracy -- Unseen Rice Dataset",
        os.path.join(output_dir, "per_class_accuracy.png"),
    )

    with open(os.path.join(output_dir, "phase1_classification_results.json"), "w") as f:
        json.dump({
            "metrics": {
                k: v for k, v in class_metrics.items()
                if k != "confusion_matrix"
            },
            "confusion_matrix": class_metrics["confusion_matrix"],
            "predictions": class_results,
        }, f, indent=2)
    print(f"  Saved: phase1_classification_results.json")

    # ===================================================================
    # PHASE 2: Multi-turn VQA with exact training-style questions
    # ===================================================================
    exact_preds, exact_refs, exact_details = run_multiturn_vqa(
        model, tokenizer, samples, EXACT_VQA_QUESTIONS,
        "Phase 2 -- Multi-turn VQA (Exact Questions)", device,
    )
    exact_agg, exact_per = compute_nlp_metrics(exact_preds, exact_refs)
    print_nlp_report(exact_agg, "Phase 2 -- Exact Questions")

    plot_per_class_nlp_metrics(
        exact_details, EXACT_VQA_QUESTIONS,
        "Exact Questions",
        os.path.join(output_dir, "per_class_nlp_exact.png"),
    )

    with open(os.path.join(output_dir, "phase2_exact_vqa_results.json"), "w") as f:
        json.dump({
            "aggregate_metrics": exact_agg,
            "per_sample_metrics": exact_per,
            "detailed_results": exact_details,
        }, f, indent=2)
    print(f"  Saved: phase2_exact_vqa_results.json")

    # ===================================================================
    # PHASE 3: Multi-turn VQA with paraphrased questions
    # ===================================================================
    para_preds, para_refs, para_details = run_multiturn_vqa(
        model, tokenizer, samples, PARAPHRASED_VQA_QUESTIONS,
        "Phase 3 -- Multi-turn VQA (Paraphrased Questions)", device,
    )
    para_agg, para_per = compute_nlp_metrics(para_preds, para_refs)
    print_nlp_report(para_agg, "Phase 3 -- Paraphrased Questions")

    plot_per_class_nlp_metrics(
        para_details, PARAPHRASED_VQA_QUESTIONS,
        "Paraphrased Questions",
        os.path.join(output_dir, "per_class_nlp_paraphrased.png"),
    )

    with open(os.path.join(output_dir, "phase3_paraphrased_vqa_results.json"), "w") as f:
        json.dump({
            "aggregate_metrics": para_agg,
            "per_sample_metrics": para_per,
            "detailed_results": para_details,
        }, f, indent=2)
    print(f"  Saved: phase3_paraphrased_vqa_results.json")

    # -- Comparison plot --
    plot_nlp_metrics_comparison(
        exact_agg, para_agg,
        os.path.join(output_dir, "nlp_metrics_comparison.png"),
    )

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    summary = {
        "timestamp": timestamp,
        "device": device,
        "base_model": BASE_MODEL_ID,
        "adapter_path": ADAPTER_PATH,
        "dataset_path": UNSEEN_DATASET_PATH,
        "total_images": len(samples),
        "per_class_count": {cls: sum(1 for s in samples if s["disease"] == cls) for cls in DISEASE_CLASSES},
        "phase1_classification": {
            "overall_accuracy": class_metrics["overall_accuracy"],
            "macro_f1": class_metrics["macro_f1"],
            "weighted_f1": class_metrics["weighted_f1"],
            "macro_precision": class_metrics["macro_precision"],
            "macro_recall": class_metrics["macro_recall"],
            "per_class_accuracy": class_metrics["per_class_accuracy"],
            "unknown_predictions": class_metrics["unknown_predictions"],
        },
        "phase2_exact_vqa": exact_agg,
        "phase3_paraphrased_vqa": para_agg,
    }

    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    summary_txt_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write("UNSEEN RICE DISEASE EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp:    {timestamp}\n")
        f.write(f"Device:       {device}\n")
        f.write(f"Base Model:   {BASE_MODEL_ID}\n")
        f.write(f"Adapter:      {ADAPTER_PATH}\n")
        f.write(f"Dataset:      {UNSEEN_DATASET_PATH}\n")
        f.write(f"Total Images: {len(samples)}\n\n")

        f.write("PHASE 1 -- Classification\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Overall Accuracy:   {class_metrics['overall_accuracy']:.4f}\n")
        f.write(f"  Macro F1:           {class_metrics['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1:        {class_metrics['weighted_f1']:.4f}\n")
        f.write(f"  Macro Precision:    {class_metrics['macro_precision']:.4f}\n")
        f.write(f"  Macro Recall:       {class_metrics['macro_recall']:.4f}\n")
        for cls, acc in class_metrics["per_class_accuracy"].items():
            f.write(f"  {cls:15s} accuracy: {acc:.4f}\n")

        f.write(f"\nPHASE 2 -- Multi-turn VQA (Exact Questions)\n")
        f.write("-" * 40 + "\n")
        for k, v in exact_agg.items():
            f.write(f"  {k:20s}: {v:.4f}\n")

        f.write(f"\nPHASE 3 -- Multi-turn VQA (Paraphrased Questions)\n")
        f.write("-" * 40 + "\n")
        for k, v in para_agg.items():
            f.write(f"  {k:20s}: {v:.4f}\n")

    print(f"\n{'=' * 70}")
    print("  EVALUATION COMPLETE")
    print(f"  All results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
