"""Model loading and saving utilities."""

import json
import logging
from pathlib import Path
from typing import Union

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def load_model(
    model_name_or_path: Union[str, Path],
    device: str = None,
) -> SentenceTransformer:
    """Load a sentence transformer model.

    Args:
        model_name_or_path: HuggingFace model name or path to saved model
        device: Device to load model on (None for auto-detect)

    Returns:
        Loaded SentenceTransformer model
    """
    model_path = Path(model_name_or_path)

    # Check if this is a LoRA model by looking for adapter_config.json
    adapter_config_path = model_path / "adapter_config.json"
    if model_path.exists() and adapter_config_path.exists():
        return _load_lora_model(model_path, device)

    # Standard loading
    model = SentenceTransformer(str(model_name_or_path), device=device)
    return model


def _load_lora_model(model_path: Path, device: str = None) -> SentenceTransformer:
    """Load a LoRA-adapted sentence transformer model.

    Args:
        model_path: Path to saved LoRA model
        device: Device to load model on

    Returns:
        Loaded SentenceTransformer model with LoRA adapters
    """
    from peft import LoraConfig, get_peft_model
    import torch

    # Load adapter config
    with open(model_path / "adapter_config.json") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config["base_model_name_or_path"]
    logger.info(f"Loading base model: {base_model_name}")

    # Load the base sentence transformer
    model = SentenceTransformer(base_model_name, device=device)

    # Get the underlying transformer
    transformer = model[0].auto_model

    # Create LoRA config
    lora_config = LoraConfig(
        r=adapter_config["r"],
        lora_alpha=adapter_config["lora_alpha"],
        lora_dropout=adapter_config.get("lora_dropout", 0.1),
        target_modules=adapter_config["target_modules"],
        bias=adapter_config.get("bias", "none"),
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(transformer, lora_config)

    # Load the saved adapter weights
    adapter_weights_path = model_path / "adapter_model.safetensors"
    if adapter_weights_path.exists():
        from safetensors.torch import load_file
        adapter_weights = load_file(str(adapter_weights_path))
    else:
        adapter_weights_path = model_path / "adapter_model.bin"
        adapter_weights = torch.load(str(adapter_weights_path), map_location="cpu")

    # Fix key naming differences between PEFT versions
    # Saved keys: ...lora_A.weight -> Expected: ...lora_A.default.weight
    fixed_weights = {}
    for key, value in adapter_weights.items():
        # Check if key needs .default. inserted
        if ".lora_A.weight" in key:
            new_key = key.replace(".lora_A.weight", ".lora_A.default.weight")
            fixed_weights[new_key] = value
        elif ".lora_B.weight" in key:
            new_key = key.replace(".lora_B.weight", ".lora_B.default.weight")
            fixed_weights[new_key] = value
        else:
            fixed_weights[key] = value

    # Load weights
    missing, unexpected = peft_model.load_state_dict(fixed_weights, strict=False)

    if missing:
        # Check if missing keys are just non-LoRA params (expected)
        lora_missing = [k for k in missing if 'lora' in k.lower()]
        if lora_missing:
            logger.warning(f"Missing LoRA keys: {lora_missing[:5]}...")
        else:
            logger.info(f"Loaded LoRA weights successfully ({len(fixed_weights)} adapter params)")
    else:
        logger.info(f"Loaded LoRA weights successfully")

    # Replace the transformer in the SentenceTransformer
    model[0].auto_model = peft_model

    # Load any other trained module weights saved alongside the adapter
    # (e.g. 2_Dense/model.safetensors). Sentence-T5 / similar models include
    # a trained Dense projection that lives outside the LoRA-wrapped
    # transformer, and was saved separately during training. Without this
    # the loader silently uses the BASE encoder's untrained Dense head.
    from safetensors.torch import load_file
    modules_json = model_path / "modules.json"
    if modules_json.exists():
        with open(modules_json) as f:
            modules_meta = json.load(f)
        for entry in modules_meta:
            sub_path = entry.get("path", "")
            if not sub_path:
                continue  # idx 0 (the Transformer) lives at root, handled above
            sub_dir = model_path / sub_path
            if not sub_dir.is_dir():
                continue
            sf = sub_dir / "model.safetensors"
            pt = sub_dir / "pytorch_model.bin"
            weights = None
            if sf.exists():
                weights = load_file(str(sf))
            elif pt.exists():
                import torch as _torch
                weights = _torch.load(str(pt), map_location="cpu")
            if weights is None:
                continue
            module_idx = entry.get("idx")
            if module_idx is None:
                continue
            sub_module = list(model._modules.values())[module_idx]
            sub_module.load_state_dict({k: v.to(device) for k, v in weights.items()}, strict=False)
            logger.info(f"Loaded trained weights for {sub_path}")

    return model


def save_model(model: SentenceTransformer, path: Union[str, Path]) -> None:
    """Save a sentence transformer model.

    Args:
        model: Model to save
        path: Directory path to save model to
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def get_device() -> str:
    """Get the best available device.

    Returns:
        Device string: 'mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' otherwise
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
