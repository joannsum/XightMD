from huggingface_hub import login
login(token="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")

from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

labels = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
]

def has_any_label(example):
    return any(example.get(label, 0) == 1 for label in labels)

ds = load_dataset(
    "BahaaEldin0/NIH-Chest-Xray-14",
    split="train",
    streaming=True
)

filtered_ds = ds.filter(has_any_label)
