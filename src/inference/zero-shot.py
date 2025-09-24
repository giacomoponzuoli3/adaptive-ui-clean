"""
Zero-Shot Image-Text-to-Text Inference Script

This module evaluates a multimodal model's ability to perform zero-shot classification
on an image-text-to-text task, using a dataset of frames paired with textual queries.

Intended Use:
-------------
This script is designed for rapid zero-shot evaluation of large multimodal models
on custom datasets, without fine-tuning, to assess baseline performance before
task-specific training.
"""

import io
import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.utils.load_dataset import get_data
from src.utils.seed import set_seed
from huggingface_hub import HfFileSystem
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig
import argparse
import random

HF_DATASET_REPO = "giacomoponzuoli3/adaptive-ui-outdoor-visibilty"

set_seed()
# To read images directly from the Hub without downloading the entire dataset
fs = HfFileSystem()  


def _map_jsonl_to_hf(sfx: str) -> str:
    """
    Adatta i path salvati nei JSONL alla struttura reale della repo HF.
    Adapts paths saved in JSONL files to the actual structure of the HF repo.
    In the JSONL files: generated_overlays/task_2/...
    In HF: task_2/...
    """
    if sfx.startswith("generated_overlays/task_2/"):
        sfx = sfx.replace("generated_overlays/task_2/", "task_2/", 1)

    return sfx

def open_image_from_hub(local_path: str) -> Image.Image:
    """Open an image directly from the HF using a local path saved in the JSONL files"""
    
    sfx = _suffix_after_data(local_path) 
    hfpath = f"hf://datasets/{HF_DATASET_REPO}/{sfx}"
    
    with fs.open(hfpath, "rb") as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def _suffix_after_data(local_path: str) -> str:
    """Return the part after '/data/' from a local path saved in the JSONL files"""

    key = local_path.split("/data/", 1)[-1]
    return _map_jsonl_to_hf(key)

def to_hf_url(local_or_suffix: str) -> str:
    # accept either a local path or already the suffix
    if "/data/" in local_or_suffix:
        suffix = _suffix_after_data(local_or_suffix)
    else:
        suffix = local_or_suffix.lstrip("/")
    return f"https://huggingface.co/datasets/{HF_DATASET_REPO}/tree/main/{suffix}"


def load_labels(jsonl_path):
    """
    Loads frame-to-label mappings from a JSONL file.

    Parameters
    ----------
    jsonl_path : str
        Path to the JSONL file containing frame file names and corresponding labels.

    Returns
    -------
    dict
        A dictionary mapping frame file names to their labels.
    """
    frame_to_label = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            raw_frame = entry["frames_file_names"][0]
            label = entry["label"]
            # get the part after /data/ 
            key = _suffix_after_data(raw_frame) 
            frame_to_label[key] = label
    return frame_to_label

def predict(model_id, dataset, labels_map, max_examples=50):
    """
    Performs zero-shot image-text-to-text inference on a dataset using a specified model.

    Parameters
    ----------
    model_id : str
        The identifier of the model to use for inference.
    dataset : list
        The dataset to evaluate, where each entry is a tuple containing metadata and input content.
    labels_map : dict
        A dictionary mapping frame file names to ground-truth labels.
    max_examples : int, optional
        The maximum number of examples to evaluate (default is 50).

    Returns
    -------
    tuple of list
        A tuple containing two lists:
        - Ground-truth labels
        - Model predictions ("yes" or "no")
    """
    labels = []
    preds = []
    examples = random.sample(dataset, k=max_examples)
    device = 0 if torch.cuda.is_available() else -1
    print(device)

    # original pipeline -> Out of Memory
    #pipe = pipeline("image-text-to-text", model=model_id, device=device)

    # Compressed model with 4-bit quantization to fit in GPU memory    
    bnb_cfg = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
    )

    # "Light" pipeline to avoid OOM
    pipe = pipeline(
        task="image-text-to-text",
        model=model_id,
        device_map="auto",                # offload automatico
        dtype=torch.bfloat16,             # nuovo arg (sostituisce torch_dtype)
        quantization_config=bnb_cfg,
        offload_folder="/content/offload" # crea cartella su disco per offload
    )
 
    for example in examples:
        # Extract original paths (from JSONL)
        original_img_path = example[1]["content"][0]["image"]  
        overlayed_img_path = example[1]["content"][1]["image"]

        # print into for debugging
        print(original_img_path)
        print(overlayed_img_path)
        
        # Label lookup via suffix consistent with load_labels
        suffix = _suffix_after_data(original_img_path)
        label = labels_map.get(suffix)
        if label is None:
            # if there's no label for this frame, skip
            continue

        # Open images directly from the Hub (no URL)
        try:
            original = open_image_from_hub(original_img_path)
            overlayed = open_image_from_hub(overlayed_img_path)
        except Exception as e:
            # If an image is missing or the path is wrong, skip the example
            print(f"[WARN] Image doesn't find in the hub for {suffix}: {e}")
            continue

        # Inject PIL.images into the example (the pipeline accepts them)
        example[1]["content"][0]["image"] = original
        example[1]["content"][1]["image"] = overlayed

        decoded = pipe(text=example, max_new_tokens=512) # tried also with 32, 16 tokens to reduce cost
        response = decoded[0]['generated_text'][2]["content"].lower()
        pred = "no" if "no, remove the element" in response else "yes"

        # Append results to lists for metrics
        labels.append(label)
        preds.append(pred)

    return labels, preds
 

def parse_args():
    """
    Parses command-line arguments for the zero-shot inference script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including dataset paths, config path, output directory, model id, and run name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="visibility")
    parser.add_argument("--train_path", type=str, default="data/train-visibility.jsonl")
    parser.add_argument("--test_path", type=str, default="data/test-visibility.jsonl")
    parser.add_argument("--val_path", type=str, default="data/val-visibility.jsonl")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--question_type", type=str, default="simple")
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task
    train_path = args.train_path
    test_path = args.test_path
    val_path = args.val_path
    model_id = args.model_id
    question_type = args.question_type

    train_dataset, test_dataset, val_dataset = get_data(task, train_path, test_path, val_path, question_type, add_label=False)
    labels_map = load_labels(train_path)
    
    # predict on train set because it's bigger than val and test
    labels, preds = predict(model_id, train_dataset, labels_map, max_examples=200)

    # metrics
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, pos_label='yes')
    precision = precision_score(labels, preds, pos_label='yes')

    model_id = model_id.split("/")[1]

    with open(f"results/zero-shot-{model_id}-{question_type}-prompt.txt", "w") as f:
        f.write("=== Evaluation Report ===\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")

if __name__ == "__main__":
    main()
 
