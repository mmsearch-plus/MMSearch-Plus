#!/usr/bin/env python3
"""
Decrypt MMSearch-Plus dataset after loading from HuggingFace Hub.

This module provides two main functions:
1. decrypt_dataset(): Decrypt an already-loaded Dataset object
2. decrypt_mmsearch_plus(): Load from path and decrypt in one step

Example usage with loaded dataset:
    from datasets import load_dataset
    from decrypt_after_load import decrypt_dataset
    
    # Load encrypted dataset
    encrypted_ds = load_dataset("Cie1/MMSearch-Plus", split='train')
    
    # Decrypt it
    decrypted_ds = decrypt_dataset(encrypted_ds, canary="MMSearch-Plus")
    
Example usage with path:
    from decrypt_after_load import decrypt_mmsearch_plus
    
    # Load and decrypt in one step
    decrypted_ds = decrypt_mmsearch_plus(
        dataset_path="Cie1/MMSearch-Plus",
        canary="MMSearch-Plus"
    )
"""

import base64
import hashlib
import argparse
import io
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset
from PIL import Image
from typing import Dict, Any
import os

def derive_key(password: str, length: int) -> bytes:
    """Derive encryption key from password using SHA-256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]

def decrypt_image(ciphertext_b64: str, password: str) -> Image.Image:
    """Decrypt base64-encoded encrypted image bytes back to PIL Image."""
    if not ciphertext_b64:
        return None

    try:
        encrypted = base64.b64decode(ciphertext_b64)
        key = derive_key(password, len(encrypted))
        decrypted = bytes([a ^ b for a, b in zip(encrypted, key)])

        # Convert bytes back to PIL Image
        img_buffer = io.BytesIO(decrypted)
        image = Image.open(img_buffer)
        return image
    except Exception as e:
        print(f"[Warning] Image decryption failed: {e}")
        return None

def decrypt_text(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext using XOR cipher with derived key."""
    if not ciphertext_b64:
        return ciphertext_b64

    try:
        encrypted = base64.b64decode(ciphertext_b64)
        key = derive_key(password, len(encrypted))
        decrypted = bytes([a ^ b for a, b in zip(encrypted, key)])
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"[Warning] Decryption failed: {e}")
        return ciphertext_b64  # Return original if decryption fails

def decrypt_sample(sample: Dict[str, Any], canary: str) -> Dict[str, Any]:
    """Decrypt text and image fields in a single sample using the provided canary password."""
    decrypted_sample = sample.copy()

    # Decrypt text fields (must match what was encrypted)
    text_fields = ['question', 'video_url', 'arxiv_id']

    for field in text_fields:
        if field in sample and sample[field]:
            decrypted_sample[field] = decrypt_text(sample[field], canary)

    # Handle answer field (list of strings)
    if 'answer' in sample and sample['answer']:
        decrypted_answers = []
        for answer in sample['answer']:
            if answer:
                decrypted_answers.append(decrypt_text(answer, canary))
            else:
                decrypted_answers.append(answer)
        decrypted_sample['answer'] = decrypted_answers

    # Images are NOT encrypted in the current version, so no image decryption needed
    # If your dataset has encrypted images (base64 strings), uncomment below:
    # image_fields = ['img_1', 'img_2', 'img_3', 'img_4', 'img_5']
    # for field in image_fields:
    #     if field in sample and sample[field] is not None and isinstance(sample[field], str):
    #         decrypted_sample[field] = decrypt_image(sample[field], canary)

    return decrypted_sample

def decrypt_dataset(encrypted_dataset: Dataset, canary: str, output_path: str = None) -> Dataset:
    """
    Decrypt an already-loaded dataset object.
    
    Args:
        encrypted_dataset: Already loaded Dataset object to decrypt
        canary: Canary string used for encryption
        output_path: Path to save decrypted dataset (optional)
    
    Returns:
        Decrypted Dataset object
    """
    if not isinstance(encrypted_dataset, Dataset):
        raise TypeError(f"Expected Dataset object, got {type(encrypted_dataset)}")
    
    print(f"📊 Dataset contains {len(encrypted_dataset)} samples")
    print(f"🔧 Features: {list(encrypted_dataset.features.keys())}")
    print(f"🔑 Using canary string: {canary}")

    # Decrypt the dataset using map function for efficiency
    print(f"🔄 Decrypting dataset...")
    
    def decrypt_batch(batch):
        """Decrypt a batch of samples."""
        # Get the number of samples in the batch
        num_samples = len(batch[list(batch.keys())[0]])
        
        # Process each sample in the batch
        decrypted_batch = {key: [] for key in batch.keys()}
        
        for i in range(num_samples):
            # Extract single sample from batch
            sample = {key: batch[key][i] for key in batch.keys()}
            
            # Decrypt sample
            decrypted_sample = decrypt_sample(sample, canary)
            
            # Add to decrypted batch
            for key in decrypted_batch.keys():
                decrypted_batch[key].append(decrypted_sample.get(key))
        
        return decrypted_batch
    
    # Apply decryption with batching
    decrypted_dataset = encrypted_dataset.map(
        decrypt_batch,
        batched=True,
        batch_size=50,
        desc="Decrypting samples"
    )

    print(f"✅ Decryption completed!")
    print(f"📝 Decrypted {len(decrypted_dataset)} samples")
    print(f"🔓 Text fields decrypted: question, answer, video_url, arxiv_id")
    print(f"🖼️ Images: kept as-is (not encrypted in current version)")
    print(f"📋 Metadata preserved: category, difficulty, subtask, etc.")

    # Save if output path provided
    if output_path:
        print(f"💾 Saving decrypted dataset to: {output_path}")
        decrypted_dataset.save_to_disk(output_path)
        print(f"✅ Saved successfully!")
    
    return decrypted_dataset

def decrypt_mmsearch_plus(dataset_path: str, canary: str, output_path: str = None, from_hub: bool = False):
    """
    Load and decrypt the MMSearch-Plus dataset.
    
    Args:
        dataset_path: Path to local dataset or HuggingFace Hub repo ID
        canary: Canary string used for encryption
        output_path: Path to save decrypted dataset (optional)
        from_hub: Whether to load from HuggingFace Hub (default: auto-detect)
    """
    # Auto-detect if loading from hub (contains "/" and doesn't exist locally)
    if not from_hub:
        from_hub = "/" in dataset_path and not Path(dataset_path).exists()
    
    # Load the encrypted dataset
    if from_hub:
        print(f"🔓 Loading encrypted dataset from HuggingFace Hub: {dataset_path}")
        # Load from HuggingFace Hub without trust_remote_code
        encrypted_dataset = load_dataset(dataset_path, split='train')
    else:
        print(f"🔓 Loading encrypted dataset from local path: {dataset_path}")
        # Check if path exists
        if not Path(dataset_path).exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        encrypted_dataset = load_from_disk(dataset_path)

    # Use decrypt_dataset to handle the actual decryption
    return decrypt_dataset(encrypted_dataset, canary, output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Decrypt MMSearch-Plus dataset after loading from HuggingFace Hub or local path.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From HuggingFace Hub
  python decrypt_after_load.py --dataset-path username/mmsearch-plus-encrypted --canary "MMSearch-Plus" --output ./decrypted
  
  # From local directory
  python decrypt_after_load.py --dataset-path ./mmsearch_plus_encrypted --canary "MMSearch-Plus" --output ./decrypted
  
  # Using environment variable for canary
  export MMSEARCH_PLUS="your-canary-string"
  python decrypt_after_load.py --dataset-path username/mmsearch-plus-encrypted --output ./decrypted
        """
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to encrypted dataset (local directory or HuggingFace Hub repo ID)"
    )
    parser.add_argument(
        "--canary",
        help="Canary string used for encryption (or set MMSEARCH_PLUS environment variable)"
    )
    parser.add_argument(
        "--output",
        help="Path to save the decrypted dataset (optional, defaults to not saving)"
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Force loading from HuggingFace Hub (auto-detected by default)"
    )
    
    args = parser.parse_args()

    # Get canary from args or environment variable
    canary = args.canary or os.environ.get("MMSEARCH_PLUS")
    
    if not canary:
        raise ValueError(
            "Canary string is required for decryption. Either provide --canary argument "
            "or set the MMSEARCH_PLUS environment variable.\n"
            "Example: export MMSEARCH_PLUS='your-canary-string'"
        )

    # Check if output path exists
    if args.output:
        output_path = Path(args.output)
        if output_path.exists():
            response = input(f"Output path {output_path} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return

    # Decrypt dataset
    decrypt_mmsearch_plus(
        dataset_path=args.dataset_path,
        canary=canary,
        output_path=args.output,
        from_hub=args.from_hub
    )

if __name__ == "__main__":
    main()

