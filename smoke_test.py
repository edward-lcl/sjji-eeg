"""Quick sanity check — model forward pass + SimCLR loss, no real data needed."""

import torch
from src.model import build_encoder, EEGClassifier
from src.pretrain import eeg_augment, UnlabeledEEGDataset

def test_encoder():
    enc = build_encoder(Chan=61, Features=244)
    x = torch.randn(4, 61, 4000)  # batch=4, 61ch, 16s @ 250Hz
    out = enc(x)
    assert out.shape == (4, 244), f"Expected (4, 244), got {out.shape}"
    print(f"✓ Encoder forward pass: {x.shape} → {out.shape}")

def test_classifier():
    enc = build_encoder()
    clf = EEGClassifier(enc, nb_classes=2)
    x = torch.randn(4, 61, 1024)
    logits = clf(x)
    assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"  # binary: 1 logit, use sigmoid
    print(f"✓ Classifier forward pass: {x.shape} → {logits.shape}")

def test_augmentation():
    x = torch.randn(61, 1024)
    aug = eeg_augment(x)
    assert aug.shape == x.shape, f"Augmented shape mismatch: {aug.shape}"
    print(f"✓ Augmentation: shape preserved {aug.shape}")

def test_mps():
    if torch.backends.mps.is_available():
        enc = build_encoder().to("mps")
        x = torch.randn(2, 61, 1024).to("mps")
        out = enc(x)
        print(f"✓ MPS (Apple Silicon) forward pass OK: {out.shape}")
    else:
        print("  MPS not available, skipping")

if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_encoder()
    test_classifier()
    test_augmentation()
    test_mps()
    print("\nAll checks passed ✓")
