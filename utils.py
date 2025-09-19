import argparse
import queue
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import sherpa_onnx
import soundfile as sf
import os
import numpy as np
import librosa
import sherpa_onnx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from operator import itemgetter

def ComputeErrorRates(scores, labels):
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    labels = [labels[i] for i in sorted_indexes]
    fnrs, fprs = [], []

    for i in range(len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])

    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm
    fnrs = [x / float(fnrs_norm) for x in fnrs]
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

def ComputeEER(fnrs, fprs, thresholds):
    min_diff = float("inf")
    eer = eer_threshold = None
    for i in range(len(fnrs)):
        diff = abs(fnrs[i] - fprs[i])
        if diff < min_diff:
            min_diff = diff
            eer = (fnrs[i] + fprs[i]) / 2
            eer_threshold = thresholds[i]
    return eer, eer_threshold

def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return min_c_det / c_def, min_c_det_threshold

def evaluate_speaker_verification(scores_intra, scores_inter, title="Speaker Verification Evaluation"):
    all_scores = np.concatenate([scores_intra, scores_inter])
    all_labels = np.concatenate([
        np.ones(len(scores_intra)),
        np.zeros(len(scores_inter))
    ])

    fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
    eer, eer_threshold = ComputeEER(fnrs, fprs, thresholds)
    mindcf, mindcf_threshold = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    plt.figure(figsize=(12, 6))
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")

    eer_threshold = round(eer_threshold, 2)
    plt.hist(scores_intra, bins=200, alpha=0.5, label="Same Speaker", color="blue")
    plt.hist(scores_inter, bins=200, alpha=0.5, label="Different Speaker", color="#FF8C00")
    plt.axvline(eer_threshold, color="red", linestyle="dashed", linewidth=1,
                label=f"EER Threshold ({eer_threshold})")
    plt.xlim(-0.5, 1)
    plt.title(f'{title} (EER: {round(eer * 100, 2)}%)')
    plt.legend()
    plt.show()

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "mindcf": mindcf,
        "mindcf_threshold": mindcf_threshold,
    }


def extract_embedding(audio, sr, extractor):
    stream = extractor.create_stream()
    stream.accept_waveform(sr, np.ascontiguousarray(audio, dtype=np.float32))
    stream.input_finished()
    return np.array(extractor.compute(stream))