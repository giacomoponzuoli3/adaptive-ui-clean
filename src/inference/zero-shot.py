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

import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.utils.load_dataset import get_data
from src.utils.seed import set_seed
from transformers import pipeline
import argparse
import random
from huggingface_hub import login
import os

HF_DATASET_REPO = "giacomoponzuoli3/adaptive-ui-outdoor-visibilty"

def _suffix_after_data(local_path: str) -> str:
    # take all after "/data/"
    key = local_path.split("/data/", 1)[-1]
    return key

def to_hf_url(local_or_suffix: str) -> str:
    # accept either a local path or already the suffix
    if "/data/" in local_or_suffix:
        suffix = _suffix_after_data(local_or_suffix)
    else:
        suffix = local_or_suffix.lstrip("/")
    return f"https://huggingface.co/datasets/{HF_DATASET_REPO}/tree/main/{suffix}"


set_seed()

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
            key = _suffix_after_data(raw_frame)   # <--- indicizza con il suffix
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
    pipe = pipeline("image-text-to-text", model=model_id, device=device)
 
    for example in examples:

        example_path = example[1]["content"][0]["image"]
        suffix = _suffix_after_data(example_path)
        label = labels_map.get(suffix)
        

        if label is None:
            print(example_path)
            print(f"Warning: No label found for {example_path}, skipping...")
            continue

        # Sostituisco l’immagine con l’URL alla Hub (streaming just-in-time)
        example[1]["content"][0]["image"] = to_hf_url(suffix)

        labels.append(label)
       
        decoded = pipe(text=example, max_new_tokens=512)
        response = decoded[0]['generated_text'][2]["content"].lower()
        pred = "no" if "no, remove the element" in response else "yes"
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

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)
    else:
        raise RuntimeError("Manca HUGGINGFACE_HUB_TOKEN")
    
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
 
