"""Shared metric conventions for the frozen sparse-coding evaluations."""

import hashlib

import numpy as np
from skimage.metrics import structural_similarity


SSIM_PROTOCOL = {
    "version": "reference-range-v1",
    "data_range": "max(reference) - min(reference); independent of reconstruction",
    "constant_reference_range": 1.0,
    "constant_reference_rule": (
        "An exactly constant reference uses data_range=1.0 in the existing "
        "input units, for every method. Neither image is rescaled or shifted."
    ),
    "parameters": {"win_size": 7, "K1": 0.01, "K2": 0.03,
                   "gaussian_weights": False, "use_sample_covariance": True},
}


def validate_frozen_manifest(observed, frozen):
    """Reject changed inputs even when the alias count happens to be unchanged."""
    for key in ['original_names', 'input_sha256_float64_le', 'aliases']:
        if observed[key] != frozen[key]:
            raise ValueError('Frozen structured inputs changed; no replacement experiment permitted')


def reference_range(reference):
    reference = np.asarray(reference)
    if reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError("SSIM reference must be nonempty and finite")
    span = float(reference.max()) - float(reference.min())
    return span if span > 0 else SSIM_PROTOCOL["constant_reference_range"]


def reference_ssim(reference, reconstruction):
    reference, reconstruction = np.asarray(reference), np.asarray(reconstruction)
    if reference.shape != reconstruction.shape:
        raise ValueError("SSIM reference and reconstruction shapes differ")
    if not np.isfinite(reconstruction).all():
        raise ValueError("SSIM reconstruction must be finite")
    return float(structural_similarity(
        reference, reconstruction, data_range=reference_range(reference),
        **SSIM_PROTOCOL["parameters"],
    ))


def exact_unique_inputs(names, patches):
    """Keep the first named representative of each exactly equal input."""
    patches = np.asarray(patches)
    if patches.ndim != 2 or len(names) != len(patches):
        raise ValueError("Expected one name per flattened input")
    if len(set(names)) != len(names) or not np.isfinite(patches).all():
        raise ValueError("Input names must be unique and values finite")
    keep, aliases = [], {}
    for i, name in enumerate(names):
        match = next((j for j in keep if np.array_equal(patches[i], patches[j])), None)
        if match is None:
            keep.append(i)
        else:
            aliases[name] = names[match]
    hashes = {
        name: hashlib.sha256(np.asarray(patch, dtype="<f8").tobytes()).hexdigest()
        for name, patch in zip(names, patches)
    }
    return np.asarray(keep, dtype=int), {
        "n_original_entries": len(names), "n_unique_inputs": len(keep),
        "aliases": aliases, "retained_names": [names[i] for i in keep],
        "original_names": list(names), "input_sha256_float64_le": hashes,
        "equality": "exact numpy.array_equal on original inputs; no tolerance",
        "inference": (
            "Retain the original batch, including aliases, to preserve the "
            "batch-relative ISTA stopping rule; score each unique input once."
        ),
    }
