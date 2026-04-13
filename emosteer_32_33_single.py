#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
import torchaudio


EMOTION_LABEL_TO_INDEX: Dict[str, int] = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "other": 5,
    "sad": 6,
    "surprised": 7,
    "unknown": 8,
}

EMOTION_ALIAS: Dict[str, str] = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgusted",
    "disgusted": "disgusted",
    "fear": "fearful",
    "fearful": "fearful",
    "happiness": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
}

EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "angry": ["angry", "anger", "愤怒", "生气"],
    "disgusted": ["disgust", "disgusted", "厌恶"],
    "fearful": ["fear", "fearful", "恐惧", "害怕"],
    "happy": ["happy", "happiness", "开心", "高兴", "快乐"],
    "neutral": ["neutral", "中立", "平静"],
    "other": ["other", "其它", "其他"],
    "sad": ["sad", "sadness", "悲伤", "伤心", "难过"],
    "surprised": ["surprise", "surprised", "惊讶"],
    "unknown": ["unknown", "未知"],
}


@dataclass
class SearchReference:
    ref_audio: str
    ref_text: str
    gen_text: str
    file_id: str


@dataclass
class RuntimeConfig:
    model_name: str
    vocoder_name: str
    device: Optional[str]
    model_cfg: Optional[str]
    ckpt_file: Optional[str]
    vocab_file: Optional[str]
    load_vocoder_from_local: bool
    f5_repo_dir: Optional[Path]


@dataclass
class RuntimeHandles:
    model: torch.nn.Module
    vocoder: torch.nn.Module
    device: str
    vocoder_name: str


@dataclass
class ExtractionConfig:
    dataset_dir: Path
    emotion: str
    neutral: str
    max_samples: int
    speaker_filter: Optional[str]
    text_mode: str
    text_seed: int
    sampling_seed: Optional[int]
    nfe_step: int
    cfg_strength: float
    sway_sampling_coef: float


@dataclass
class BuildConfig:
    top_k: int
    target_emotion: str
    search_samples: int
    nfe_step: int
    cfg_strength: float
    sway_sampling_coef: float
    sampling_seed: Optional[int]
    emotion2vec_model: str
    emotion2vec_hub: str
    step_aggregation_mode: str
    post_agg_norm: bool


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("emosteer3233")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").strip()


def infer_dataset_tag(dataset_dir: Path) -> str:
    name = dataset_dir.name.strip().lower()
    if not name:
        return ""
    if name in {"dataset_esd_sorted", "esd"} or "esd" in name:
        return "esd"
    if "emilia" in name:
        return "emo_emilia"
    return safe_name(name)


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def parse_speaker_filter(speaker_filter: Optional[str]) -> Optional[Set[str]]:
    if not speaker_filter:
        return None
    result: Set[str] = set()
    if "-" in speaker_filter:
        left, right = speaker_filter.split("-", maxsplit=1)
        start, end = int(left), int(right)
        for sid in range(start, end + 1):
            result.add(f"{sid:04d}")
    else:
        for sid in speaker_filter.split(","):
            sid = sid.strip()
            if sid:
                result.add(sid)
    return result if result else None


def speaker_id_from_file(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[0] if parts else path.stem


def balanced_sample_by_speaker(files: Sequence[Path], max_samples: int, seed: Optional[int]) -> List[Path]:
    if max_samples <= 0 or len(files) <= max_samples:
        return list(files)
    rng = random.Random(seed)
    by_speaker: Dict[str, List[Path]] = {}
    for wav in files:
        by_speaker.setdefault(speaker_id_from_file(wav), []).append(wav)

    speaker_ids = sorted(by_speaker.keys())
    rng.shuffle(speaker_ids)
    for sid in speaker_ids:
        rng.shuffle(by_speaker[sid])

    picked: List[Path] = []
    cursor = 0
    active = list(speaker_ids)
    while active and len(picked) < max_samples:
        sid = active[cursor % len(active)]
        bucket = by_speaker[sid]
        if bucket:
            picked.append(bucket.pop())
            cursor += 1
            continue
        active.remove(sid)
        if active:
            cursor %= len(active)
    return picked


def collect_audio_files(
    dataset_dir: Path,
    emotion_subdir: str,
    speaker_filter: Optional[Set[str]],
    max_samples: int,
    sample_seed: Optional[int],
) -> List[Path]:
    base = dataset_dir / emotion_subdir
    files = sorted(base.rglob("*.wav"))
    if speaker_filter:
        files = [x for x in files if speaker_id_from_file(x) in speaker_filter]
    if max_samples > 0:
        files = balanced_sample_by_speaker(files, max_samples=max_samples, seed=sample_seed)
    return files


def load_transcription_map(dataset_dir: Path) -> Dict[str, str]:
    tdir = dataset_dir / "transcription"
    mapping: Dict[str, str] = {}
    if not tdir.exists():
        return mapping
    for txt in sorted(tdir.glob("*.txt")):
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def build_text_pools(trans_map: Dict[str, str], seed: int) -> Dict[str, List[str]]:
    pools = {"all": [], "en": [], "zh": []}
    seen: Set[str] = set()
    for _, text in sorted(trans_map.items()):
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        pools["all"].append(text)
        pools["zh" if contains_cjk(text) else "en"].append(text)
    rng = random.Random(seed)
    for k in pools:
        rng.shuffle(pools[k])
    return pools


def select_generation_text(
    text_pools: Dict[str, List[str]],
    sample_idx: int,
    ref_text: str,
    fallback_text: str = "This is a sample text for activation extraction.",
) -> str:
    preferred: Optional[List[str]] = None
    if contains_cjk(ref_text) and text_pools["zh"]:
        preferred = text_pools["zh"]
    elif text_pools["en"]:
        preferred = text_pools["en"]
    elif text_pools["all"]:
        preferred = text_pools["all"]

    if not preferred:
        return ref_text or fallback_text

    start = sample_idx % len(preferred)
    for offset in range(len(preferred)):
        cand = preferred[(start + offset) % len(preferred)]
        if cand and cand != ref_text:
            return cand
    return preferred[start]


def ensure_f5_tts_importable(f5_repo_dir: Optional[Path]) -> None:
    if f5_repo_dir is None:
        return
    src_dir = f5_repo_dir / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"未找到 f5_tts 源码目录: {src_dir}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_runtime(config: RuntimeConfig, logger: logging.Logger) -> RuntimeHandles:
    ensure_f5_tts_importable(config.f5_repo_dir)

    from cached_path import cached_path
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_model, load_vocoder

    device = resolve_device(config.device)

    cfg_file = config.model_cfg
    if not cfg_file:
        try_names = [f"{config.model_name}.yaml", "F5TTS_v1_Base.yaml", "F5TTS_Base.yaml"]
        for name in try_names:
            traversable = files("f5_tts").joinpath(f"configs/{name}")
            try:
                if traversable.is_file():
                    cfg_file = str(traversable)
                    break
            except Exception:
                continue
        if not cfg_file:
            raise FileNotFoundError("未找到模型配置文件，请通过 --model_cfg 指定。")

    model_cfg = OmegaConf.load(cfg_file)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    ckpt_file = config.ckpt_file
    if not ckpt_file:
        repo_name = "F5-TTS"
        ckpt_model_name = config.model_name
        ckpt_step = 1_250_000
        ckpt_type = "safetensors"
        if config.model_name == "F5TTS_Base":
            ckpt_step = 1_200_000
            if config.vocoder_name == "bigvgan":
                ckpt_model_name = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif config.model_name == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1_200_000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{ckpt_model_name}/model_{ckpt_step}.{ckpt_type}"))

    if config.vocoder_name == "vocos":
        vocoder_local_path = str(Path("checkpoints") / "vocos-mel-24khz")
    else:
        vocoder_local_path = str(Path("checkpoints") / "bigvgan_v2_24khz_100band_256x")

    logger.info(f"[运行时] 加载声码器: {config.vocoder_name}")
    vocoder = load_vocoder(
        vocoder_name=config.vocoder_name,
        is_local=config.load_vocoder_from_local,
        local_path=vocoder_local_path,
        device=device,
    )

    logger.info(f"[运行时] 加载模型: {config.model_name}")
    model = load_model(
        model_cls=model_cls,
        model_cfg=model_arc,
        ckpt_path=ckpt_file,
        mel_spec_type=config.vocoder_name,
        vocab_file=config.vocab_file or "",
        device=device,
    )
    model.eval()
    try:
        vocoder.eval()
    except Exception:
        pass
    return RuntimeHandles(model=model, vocoder=vocoder, device=device, vocoder_name=config.vocoder_name)


def parse_layers(layer_arg: str, total_layers: int, model_name: str) -> List[int]:
    if layer_arg == "all":
        return list(range(total_layers))
    if layer_arg == "paper":
        if model_name.startswith("E2TTS"):
            return [i for i in range(1, total_layers, 3)]
        return [i for i in range(1, total_layers, 5)]
    out = [int(x.strip()) for x in layer_arg.split(",") if x.strip()]
    return [x for x in out if 0 <= x < total_layers]


def normalize_emotion_label(label: str) -> str:
    return EMOTION_ALIAS.get(label.lower(), "happy")


def normalize_label_text(text: str) -> str:
    return str(text).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def resolve_target_index_from_labels(labels: Sequence[str], canonical_emotion: str) -> int:
    if not labels:
        return EMOTION_LABEL_TO_INDEX.get(canonical_emotion, 3)
    nlabels = [normalize_label_text(x) for x in labels]
    keywords = [normalize_label_text(x) for x in EMOTION_KEYWORDS.get(canonical_emotion, [canonical_emotion])]
    for i, lbl in enumerate(nlabels):
        for kw in keywords:
            if kw and kw in lbl:
                return i
    return EMOTION_LABEL_TO_INDEX.get(canonical_emotion, 3)


def load_emotion2vec_model(model_name: str, hub: str, logger: logging.Logger):
    from funasr import AutoModel

    logger.info(f"[Emotion2Vec] 加载模型: {model_name} (hub={hub})")
    return AutoModel(model=model_name, hub=hub)


def patch_ditblock_forward_if_needed(model: torch.nn.Module, logger: logging.Logger) -> None:
    blocks = getattr(getattr(model, "transformer", None), "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("模型结构不包含 transformer.transformer_blocks，无法注入 hook。")

    patched = 0
    for idx, block in enumerate(blocks):
        if getattr(block, "_emosteer_hooked", False):
            continue
        required = ["attn_norm", "attn", "ff_norm", "ff"]
        if not all(hasattr(block, n) for n in required):
            continue

        # 参照论文附录：在 DiTBlock 内部对 block 输入施加 step steering，
        # 并在第一残差 x_res1 处捕获激活。
        def patched_forward(self, x, t, mask=None, rope=None):
            ref_audio_len = getattr(self, "current_ref_audio_len", None)
            if ref_audio_len is None:
                ref_audio_len = x.shape[1]
            ref_audio_len = max(0, min(int(ref_audio_len), int(x.shape[1])))

            cond_batch = getattr(self, "cfg_batch_size", None)
            if cond_batch is None:
                cond_batch = x.shape[0]
            cond_batch = max(1, min(int(cond_batch), int(x.shape[0])))

            step_steering = getattr(self, "step_steering", None)
            if step_steering is not None and ref_audio_len > 0:
                if not torch.is_tensor(step_steering):
                    raise RuntimeError("step_steering 不是张量，无法执行 token 注入。")
                if step_steering.ndim == 1:
                    act = step_steering
                elif step_steering.ndim == 2:
                    call_counter = int(getattr(self, "_step_call_counter", 0))
                    raw_step_idx = getattr(self, "current_step_idx", None)
                    if raw_step_idx is None:
                        raw_step_idx = call_counter
                    step_idx = max(0, min(int(raw_step_idx), int(step_steering.shape[0]) - 1))
                    act = step_steering[step_idx]
                    self._step_call_counter = call_counter + 1
                else:
                    raise RuntimeError(f"step_steering 维度不支持: {tuple(step_steering.shape)}")

                act = act.to(device=x.device, dtype=x.dtype)
                act = act / (act.norm(p=2) + 1e-8)
                act = act.unsqueeze(0).repeat(ref_audio_len, 1).unsqueeze(0)  # [1, ref_len, D]

                pad_len = x.shape[1] - ref_audio_len
                if pad_len > 0:
                    pad = torch.zeros(1, pad_len, x.shape[2], dtype=x.dtype, device=x.device)
                    act = torch.cat([act, pad], dim=1)

                steer = torch.zeros_like(x)
                steer[:cond_batch] = act.expand(cond_batch, -1, -1)
                alpha = float(getattr(self, "alpha", 1.0))

                orig_norm = x.norm(p=2, dim=(1, 2), keepdim=True)
                x = x + alpha * steer
                new_norm = x.norm(p=2, dim=(1, 2), keepdim=True) + 1e-8
                x = x * (orig_norm / new_norm)

            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
            attn_output = self.attn(x=norm, mask=mask, rope=rope)

            x_res1 = x + gate_msa.unsqueeze(1) * attn_output

            if bool(getattr(self, "save_residual", False)):
                self.first_residual = x_res1.detach()
                if ref_audio_len > 0:
                    captured = x_res1[0, :ref_audio_len, :].detach().cpu()
                    total_steps = getattr(self, "current_total_steps", None)
                    capture_counter = int(getattr(self, "_capture_call_counter", 0))
                    cap_idx = getattr(self, "current_step_idx", None)
                    if cap_idx is None:
                        cap_idx = capture_counter
                    self._capture_call_counter = capture_counter + 1

                    if total_steps is not None and int(total_steps) > 0:
                        total_steps = int(total_steps)
                        if getattr(self, "step_residual_tokens", None) is None or len(self.step_residual_tokens) != total_steps:
                            self.step_residual_tokens = [None] * total_steps
                        cap_idx = max(0, min(int(cap_idx), total_steps - 1))
                        self.step_residual_tokens[cap_idx] = captured
                    else:
                        if getattr(self, "step_residual_tokens", None) is None:
                            self.step_residual_tokens = []
                        self.step_residual_tokens.append(captured)

            norm2 = self.ff_norm(x_res1) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm2)
            x_res2 = x_res1 + gate_mlp.unsqueeze(1) * ff_output
            return x_res2

        block.forward = MethodType(patched_forward, block)
        block._emosteer_hooked = True
        block.save_residual = False
        block.first_residual = None
        block.alpha = 0.0
        block.step_steering = None
        block.current_step_idx = None
        block.current_total_steps = None
        block.current_ref_audio_len = None
        block.cfg_batch_size = None
        block.step_residual_tokens = None
        block._step_call_counter = 0
        block._capture_call_counter = 0
        patched += 1

    if patched == 0:
        logger.warning("[Hook] 未找到可 patch 的 DiTBlock，若模型本身不支持 residual capture，后续会失败。")
    else:
        logger.info(f"[Hook] 已 patch DiTBlock 数量: {patched}")


def reset_all_blocks(model: torch.nn.Module) -> None:
    blocks = model.transformer.transformer_blocks
    for block in blocks:
        block.save_residual = False
        block.alpha = 0.0
        block.step_steering = None
        block.current_step_idx = None
        block.current_total_steps = None
        block.current_ref_audio_len = None
        block.cfg_batch_size = None
        block.step_residual_tokens = None
        block._step_call_counter = 0
        block._capture_call_counter = 0


def enable_residual_capture(model: torch.nn.Module, selected_layers: Sequence[int], nfe_step: int) -> None:
    reset_all_blocks(model)
    for layer_idx in selected_layers:
        block = model.transformer.transformer_blocks[layer_idx]
        block.save_residual = True
        block.step_residual_tokens = [None] * int(nfe_step)
        block.current_total_steps = int(nfe_step)
        block._capture_call_counter = 0


def disable_residual_capture(model: torch.nn.Module) -> None:
    for block in model.transformer.transformer_blocks:
        block.save_residual = False


def clear_token_steering(model: torch.nn.Module, selected_layers: Sequence[int]) -> None:
    for layer_idx in selected_layers:
        block = model.transformer.transformer_blocks[layer_idx]
        block.step_steering = None
        block.alpha = 0.0
        block._step_call_counter = 0


def set_token_steering(
    model: torch.nn.Module,
    selected_layers: Sequence[int],
    per_layer_token_step_vecs: Sequence[torch.Tensor],
    alpha: float,
    nfe_step: int,
    ref_audio_len: int,
) -> None:
    if len(selected_layers) != len(per_layer_token_step_vecs):
        raise ValueError("selected_layers 与 per_layer_token_step_vecs 长度不一致。")
    model_dtype = next(model.parameters()).dtype
    for layer_idx, vec in zip(selected_layers, per_layer_token_step_vecs):
        block = model.transformer.transformer_blocks[layer_idx]
        block.step_steering = vec.to(device=next(model.parameters()).device, dtype=model_dtype)
        block.alpha = float(alpha)
        block.current_total_steps = int(nfe_step)
        block.current_ref_audio_len = int(ref_audio_len)
        block._step_call_counter = 0


def set_runtime_context_for_all_blocks(
    model: torch.nn.Module,
    nfe_step: int,
    ref_audio_len: int,
    cfg_batch_size: int,
) -> None:
    for block in model.transformer.transformer_blocks:
        block.current_total_steps = int(nfe_step)
        block.current_ref_audio_len = int(ref_audio_len)
        block.cfg_batch_size = int(cfg_batch_size)
        block.current_step_idx = None


def estimate_ref_audio_token_len(wav_path: str, hop_length: int = 256) -> int:
    try:
        info = torchaudio.info(str(wav_path))
        return max(1, int(info.num_frames) // int(hop_length))
    except Exception:
        return 1


def fill_missing_steps(
    step_tokens: Sequence[Optional[torch.Tensor]],
    target_steps: int,
) -> Optional[List[torch.Tensor]]:
    if not step_tokens:
        return None

    first_valid = next((x for x in step_tokens if x is not None), None)
    if first_valid is None:
        return None

    dense: List[torch.Tensor] = []
    last = first_valid
    for item in step_tokens:
        if item is None:
            dense.append(last)
        else:
            last = item
            dense.append(item)
    if not dense:
        return None

    if len(dense) == target_steps:
        return dense

    # 当内部求解器调用次数与 nfe_step 不完全一致时，做均匀重采样对齐。
    if target_steps <= 1:
        return [dense[-1]]

    idx = torch.linspace(0, len(dense) - 1, target_steps).round().to(torch.long).tolist()
    return [dense[i] for i in idx]


def match_token_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[1] == target_len:
        return x
    return F.interpolate(x.permute(0, 2, 1), size=target_len, mode="nearest").permute(0, 2, 1)


def apply_step_aggregation_mode(step_steer: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "per_step":
        return step_steer
    if mode == "mean_repeat":
        mean_vec = step_steer.mean(dim=0, keepdim=True)
        return mean_vec.repeat(step_steer.shape[0], 1)
    raise ValueError(f"不支持的 step 聚合模式: {mode}")


def build_extract_meta(
    dataset_dir: Path,
    selected_layers: Sequence[int],
    audio_files: Sequence[Path],
    cfg: ExtractionConfig,
    target_len_source: str,
) -> Dict[str, Any]:
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_tag": infer_dataset_tag(dataset_dir),
        "selected_layers": [int(x) for x in selected_layers],
        "num_input_files": int(len(audio_files)),
        "selected_file_ids": [x.stem for x in audio_files],
        "max_samples": int(cfg.max_samples),
        "speaker_filter": cfg.speaker_filter,
        "text_mode": cfg.text_mode,
        "text_seed": int(cfg.text_seed),
        "sampling_seed": int(cfg.sampling_seed) if cfg.sampling_seed is not None else None,
        "nfe_step": int(cfg.nfe_step),
        "cfg_strength": float(cfg.cfg_strength),
        "sway_sampling_coef": float(cfg.sway_sampling_coef),
        "target_len_source": target_len_source,
    }


def estimate_target_len_from_captured(
    runtime: RuntimeHandles,
    emotion_subdir: str,
    selected_layers: Sequence[int],
    audio_files: Sequence[Path],
    text_mode: str,
    text_seed: int,
    sampling_seed: Optional[int],
    nfe_step: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    transcription_map: Dict[str, str],
    logger: logging.Logger,
) -> int:
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    if not audio_files:
        raise RuntimeError(f"[提取] 情绪目录 '{emotion_subdir}' 没有 wav 文件。")

    text_pools = build_text_pools(transcription_map, seed=text_seed)
    enable_residual_capture(runtime.model, selected_layers, nfe_step=nfe_step)

    captured_lens: List[int] = []
    logger.info(f"[提取] 估计目标 token 长度 | 情绪={emotion_subdir} | 样本数={len(audio_files)}")

    try:
        for i, wav_path in enumerate(audio_files, start=1):
            file_id = wav_path.stem
            ref_text = transcription_map.get(file_id, "")
            if text_mode == "random_pool":
                gen_text = select_generation_text(text_pools, i - 1, ref_text)
            else:
                gen_text = ref_text or "This is a sample text for activation extraction."

            for layer_idx in selected_layers:
                block = runtime.model.transformer.transformer_blocks[layer_idx]
                block.step_residual_tokens = [None] * int(nfe_step)
                block._capture_call_counter = 0

            ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(str(wav_path), ref_text)
            ref_audio_len = estimate_ref_audio_token_len(str(ref_audio_processed))
            set_runtime_context_for_all_blocks(runtime.model, nfe_step=nfe_step, ref_audio_len=ref_audio_len, cfg_batch_size=1)

            infer_process(
                ref_audio=ref_audio_processed,
                ref_text=ref_text_processed,
                gen_text=gen_text,
                model_obj=runtime.model,
                vocoder=runtime.vocoder,
                mel_spec_type=runtime.vocoder_name,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                device=runtime.device,
                seed=(sampling_seed + i - 1) if sampling_seed is not None else None,
            )

            sample_len: Optional[int] = None
            for layer_idx in selected_layers:
                tokens = fill_missing_steps(runtime.model.transformer.transformer_blocks[layer_idx].step_residual_tokens or [], nfe_step)
                if not tokens:
                    continue
                if tokens[0].ndim >= 2 and tokens[0].shape[0] > 0:
                    sample_len = int(tokens[0].shape[0])
                    break

            if sample_len is not None:
                captured_lens.append(sample_len)
            if i == 1 and sample_len is None:
                raise RuntimeError("第一条样本没有捕获到第一残差，hook 可能未生效。")
            if i % 20 == 0 or i == len(audio_files):
                logger.info(f"[提取] 目标长度估计进度: {i}/{len(audio_files)}")
    finally:
        disable_residual_capture(runtime.model)

    if not captured_lens:
        raise RuntimeError(f"[提取] 未能估计目标 token 长度: {emotion_subdir}")

    avg_len = max(1, int(round(sum(captured_lens) / len(captured_lens))))
    logger.info(
        f"[提取] 目标 token 长度估计完成 | 情绪={emotion_subdir} | 平均={avg_len} | "
        f"样本数={len(captured_lens)} | 最小={min(captured_lens)} | 最大={max(captured_lens)}"
    )
    return avg_len


def extract_mean_activation(
    runtime: RuntimeHandles,
    dataset_dir: Path,
    emotion_subdir: str,
    selected_layers: Sequence[int],
    audio_files: Sequence[Path],
    target_len: int,
    target_len_source: str,
    cfg: ExtractionConfig,
    transcription_map: Dict[str, str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    if not audio_files:
        raise RuntimeError(f"[提取] '{emotion_subdir}' 无可用样本。")

    text_pools = build_text_pools(transcription_map, seed=cfg.text_seed)
    enable_residual_capture(runtime.model, selected_layers, nfe_step=cfg.nfe_step)

    layer_sums: List[Optional[torch.Tensor]] = [None for _ in selected_layers]
    layer_counts: List[int] = [0 for _ in selected_layers]

    logger.info(
        f"[提取] 开始提取第一残差均值 | emotion={emotion_subdir} | 样本={len(audio_files)} | "
        f"目标长度={target_len}({target_len_source})"
    )

    try:
        for i, wav_path in enumerate(audio_files, start=1):
            file_id = wav_path.stem
            ref_text = transcription_map.get(file_id, "")
            if cfg.text_mode == "random_pool":
                gen_text = select_generation_text(text_pools, i - 1, ref_text)
            else:
                gen_text = ref_text or "This is a sample text for activation extraction."

            for layer_idx in selected_layers:
                block = runtime.model.transformer.transformer_blocks[layer_idx]
                block.step_residual_tokens = [None] * int(cfg.nfe_step)
                block._capture_call_counter = 0

            ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(str(wav_path), ref_text)
            ref_audio_len = estimate_ref_audio_token_len(str(ref_audio_processed))
            set_runtime_context_for_all_blocks(runtime.model, nfe_step=cfg.nfe_step, ref_audio_len=ref_audio_len, cfg_batch_size=1)

            infer_process(
                ref_audio=ref_audio_processed,
                ref_text=ref_text_processed,
                gen_text=gen_text,
                model_obj=runtime.model,
                vocoder=runtime.vocoder,
                mel_spec_type=runtime.vocoder_name,
                nfe_step=cfg.nfe_step,
                cfg_strength=cfg.cfg_strength,
                sway_sampling_coef=cfg.sway_sampling_coef,
                device=runtime.device,
                seed=(cfg.sampling_seed + i - 1) if cfg.sampling_seed is not None else None,
            )

            for local_idx, layer_idx in enumerate(selected_layers):
                block = runtime.model.transformer.transformer_blocks[layer_idx]
                tokens = fill_missing_steps(block.step_residual_tokens or [], cfg.nfe_step)
                if tokens is None:
                    continue
                activation = torch.stack(tokens, dim=0).to(torch.float32)  # [steps, ref_len, d]
                if target_len > 0 and activation.shape[1] != target_len:
                    activation = match_token_len(activation, target_len)
                if not torch.isfinite(activation).all():
                    logger.warning(f"[提取] 非有限值，跳过样本 {wav_path.name} @ layer={layer_idx}")
                    continue
                if layer_sums[local_idx] is None:
                    layer_sums[local_idx] = activation
                else:
                    layer_sums[local_idx] += activation
                layer_counts[local_idx] += 1

            if i == 1 and not any(c > 0 for c in layer_counts):
                raise RuntimeError("[提取] 第一条样本未捕获到任何第一残差，停止执行。")
            if i % 10 == 0 or i == len(audio_files):
                logger.info(f"[提取] 进度 {emotion_subdir}: {i}/{len(audio_files)}")
    finally:
        disable_residual_capture(runtime.model)

    if not any(c > 0 for c in layer_counts):
        raise RuntimeError(f"[提取] {emotion_subdir} 完全未捕获到残差。")

    mean_residuals: List[Optional[torch.Tensor]] = []
    for s, c in zip(layer_sums, layer_counts):
        if s is None or c == 0:
            mean_residuals.append(None)
            continue
        m = s / c
        if not torch.isfinite(m).all():
            logger.warning("[提取] 均值中出现非有限值，丢弃该层。")
            mean_residuals.append(None)
        else:
            mean_residuals.append(m.cpu())

    meta = build_extract_meta(
        dataset_dir=dataset_dir,
        selected_layers=selected_layers,
        audio_files=audio_files,
        cfg=cfg,
        target_len_source=target_len_source,
    )

    pack = {
        "layers": list(selected_layers),
        "mean_residuals": mean_residuals,
        "layer_counts": layer_counts,
        "num_samples": max(layer_counts) if layer_counts else 0,
        "target_len": int(target_len),
        "emotion": emotion_subdir,
        "neutral": cfg.neutral,
        "meta": meta,
    }
    logger.info(f"[提取] 完成 emotion={emotion_subdir} | layer_counts={dict(zip(selected_layers, layer_counts))}")
    return pack


def build_search_references(
    dataset_dir: Path,
    neutral_subdir: str,
    transcription_map: Dict[str, str],
    num_refs: int,
    speaker_filter: Optional[Set[str]],
    exclude_file_ids: Optional[Sequence[str]],
    seed: int,
    logger: logging.Logger,
) -> List[SearchReference]:
    all_neutral = collect_audio_files(
        dataset_dir=dataset_dir,
        emotion_subdir=neutral_subdir,
        speaker_filter=speaker_filter,
        max_samples=0,
        sample_seed=None,
    )
    if not all_neutral:
        return []

    exclude_ids = {str(x) for x in (exclude_file_ids or []) if str(x)}
    if exclude_ids:
        before = len(all_neutral)
        all_neutral = [x for x in all_neutral if x.stem not in exclude_ids]
        logger.info(f"[搜索样本] neutral池: 原始={before} | 排除={len(exclude_ids)} | 剩余={len(all_neutral)}")

    if len(all_neutral) < num_refs:
        raise RuntimeError(
            f"[搜索样本] 可用 neutral 数量不足: 当前={len(all_neutral)} < search_samples={num_refs}. "
            "可减小 search_samples 或 max_samples，或放宽 speaker_filter。"
        )

    pools = build_text_pools(transcription_map, seed=seed)
    rng = random.Random(seed)

    refs_with_lang: List[Dict[str, str]] = []
    for wav in all_neutral:
        fid = wav.stem
        ref_text = transcription_map.get(fid, "")
        refs_with_lang.append(
            {
                "file_id": fid,
                "ref_audio": str(wav),
                "ref_text": ref_text,
                "lang": "zh" if contains_cjk(ref_text) else "en",
            }
        )

    rng.shuffle(refs_with_lang)
    en_refs = [x for x in refs_with_lang if x["lang"] == "en"]
    zh_refs = [x for x in refs_with_lang if x["lang"] == "zh"]

    selected: List[Dict[str, str]] = []
    half = num_refs // 2
    if en_refs and zh_refs:
        selected.extend(en_refs[:half])
        selected.extend(zh_refs[: max(0, num_refs - len(selected))])
    else:
        selected.extend(refs_with_lang[:num_refs])

    if len(selected) < num_refs:
        seen = {x["file_id"] for x in selected}
        for item in refs_with_lang:
            if item["file_id"] in seen:
                continue
            selected.append(item)
            if len(selected) >= num_refs:
                break

    out: List[SearchReference] = []
    for i, item in enumerate(selected[:num_refs]):
        out.append(
            SearchReference(
                ref_audio=item["ref_audio"],
                ref_text=item["ref_text"],
                gen_text=select_generation_text(pools, i, item["ref_text"]),
                file_id=item["file_id"],
            )
        )
    logger.info(f"[搜索样本] 已构建 emotion2vec 搜索样本: {len(out)}")
    return out


def evaluate_tokens_with_emotion2vec(
    runtime: RuntimeHandles,
    ser_model: Any,
    layer_steering_vectors: Sequence[torch.Tensor],
    selected_layers: Sequence[int],
    target_emotion: str,
    references: Sequence[SearchReference],
    nfe_step: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    sampling_seed: Optional[int],
    logger: logging.Logger,
) -> torch.Tensor:
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    if not references:
        raise ValueError("emotion2vec 搜索样本不能为空。")

    canonical = normalize_emotion_label(target_emotion)
    target_idx = EMOTION_LABEL_TO_INDEX[canonical]
    target_idx_resolved = False

    normalized_vectors: List[torch.Tensor] = []
    for vec in layer_steering_vectors:
        vec = vec.to(runtime.device)
        normalized_vectors.append(vec / (vec.norm(p=2) + 1e-8))

    steps, num_tokens = normalized_vectors[0].shape[:2]
    for vec in normalized_vectors[1:]:
        if vec.shape[:2] != (steps, num_tokens):
            raise ValueError("各层 steering shape 不一致。")

    processed_refs: List[Dict[str, Any]] = []
    for ref in references:
        ref_audio_path, ref_text = preprocess_ref_audio_text(ref.ref_audio, ref.ref_text)
        processed_refs.append(
            {
                "ref_audio": ref_audio_path,
                "ref_text": ref_text,
                "gen_text": ref.gen_text,
                "file_id": ref.file_id,
                "ref_audio_len": estimate_ref_audio_token_len(str(ref_audio_path)),
            }
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="emosteer_ser_"))
    token_scores: List[float] = []
    logger.info(f"[Token评分] emotion2vec 全 token 搜索开始 | tokens={num_tokens} | refs={len(processed_refs)}")

    try:
        for token_idx in range(num_tokens):
            single_token_scores: List[float] = []
            model_dtype = next(runtime.model.parameters()).dtype
            token_vecs = [layer[:, token_idx, :].to(dtype=model_dtype) for layer in normalized_vectors]

            for ref_idx, ref in enumerate(processed_refs):
                set_runtime_context_for_all_blocks(
                    runtime.model,
                    nfe_step=nfe_step,
                    ref_audio_len=int(ref["ref_audio_len"]),
                    cfg_batch_size=1,
                )
                set_token_steering(
                    runtime.model,
                    selected_layers=selected_layers,
                    per_layer_token_step_vecs=token_vecs,
                    alpha=1.0,
                    nfe_step=nfe_step,
                    ref_audio_len=int(ref["ref_audio_len"]),
                )
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        wav_np, sample_rate, _ = infer_process(
                            ref_audio=ref["ref_audio"],
                            ref_text=ref["ref_text"],
                            gen_text=ref["gen_text"],
                            model_obj=runtime.model,
                            vocoder=runtime.vocoder,
                            mel_spec_type=runtime.vocoder_name,
                            show_info=lambda *_args, **_kwargs: None,
                            progress=None,
                            nfe_step=nfe_step,
                            cfg_strength=cfg_strength,
                            sway_sampling_coef=sway_sampling_coef,
                            device=runtime.device,
                            seed=sampling_seed,
                        )
                finally:
                    clear_token_steering(runtime.model, selected_layers)

                if wav_np is None:
                    single_token_scores.append(0.0)
                    continue

                wav_path = temp_dir / f"token_{token_idx:04d}_ref_{ref_idx:02d}.wav"
                save_wav = torch.from_numpy(wav_np).to(torch.float32)
                if save_wav.ndim == 1:
                    save_wav = save_wav.unsqueeze(0)
                torchaudio.save(str(wav_path), save_wav.cpu(), int(sample_rate))

                ser_result = ser_model.generate(str(wav_path), granularity="utterance", extract_embedding=False)
                if ser_result and len(ser_result) > 0:
                    labels = ser_result[0].get("labels", [])
                    if not target_idx_resolved:
                        target_idx = resolve_target_index_from_labels(labels, canonical)
                        mapped = labels[target_idx] if labels and len(labels) > target_idx else f"idx={target_idx}"
                        logger.info(f"[Token评分] 目标情绪 '{canonical}' 映射到标签 '{mapped}' (index={target_idx})")
                        target_idx_resolved = True
                    scores = ser_result[0].get("scores", [])
                    single_token_scores.append(float(scores[target_idx]) if len(scores) > target_idx else 0.0)
                else:
                    single_token_scores.append(0.0)

            token_scores.append(sum(single_token_scores) / max(1, len(single_token_scores)))
            if (token_idx + 1) % 10 == 0 or token_idx == num_tokens - 1:
                logger.info(
                    f"[Token评分] 进度 {token_idx + 1}/{num_tokens} | "
                    f"当前token均分={token_scores[-1]:.6f}"
                )
    finally:
        clear_token_steering(runtime.model, selected_layers)
        shutil.rmtree(temp_dir, ignore_errors=True)

    return torch.tensor(token_scores, dtype=torch.float32, device=runtime.device)


def load_residual_pack(path: Path) -> Dict[str, Any]:
    pack = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(pack, dict) or "mean_residuals" not in pack:
        raise ValueError(f"残差文件格式非法: {path}")
    return pack


def build_steering_bundle(
    runtime: RuntimeHandles,
    neutral_residual_file: Path,
    emotion_residual_file: Path,
    output_file: Path,
    cfg: BuildConfig,
    dataset_dir: Path,
    neutral_label: str,
    speaker_filter: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    neutral_pack = load_residual_pack(neutral_residual_file)
    emotion_pack = load_residual_pack(emotion_residual_file)

    neutral_layers = neutral_pack["layers"]
    emotion_layers = emotion_pack["layers"]
    if neutral_layers != emotion_layers:
        raise RuntimeError("neutral 与 emotion 的层列表不一致。")

    active_layers: List[int] = []
    unnormalized_vectors: List[torch.Tensor] = []

    logger.info("[构建] 开始构建 steering 向量")
    for layer_idx, n, e in zip(neutral_layers, neutral_pack["mean_residuals"], emotion_pack["mean_residuals"]):
        if n is None or e is None:
            logger.warning(f"[构建] 跳过层 {layer_idx}: neutral/emotion 存在 None")
            continue
        if n.shape[0] != e.shape[0] or n.shape[2] != e.shape[2]:
            raise RuntimeError(f"[构建] 层 {layer_idx} 形状不匹配: {tuple(n.shape)} vs {tuple(e.shape)}")

        target_len = (int(n.shape[1]) + int(e.shape[1])) // 2
        n = match_token_len(n, target_len)
        e = match_token_len(e, target_len)
        ul = e - n

        if not (torch.isfinite(n).all() and torch.isfinite(e).all() and torch.isfinite(ul).all()):
            logger.warning(f"[构建] 跳过层 {layer_idx}: 包含非有限值")
            continue

        unnormalized_vectors.append(ul)
        active_layers.append(int(layer_idx))
        logger.info(
            f"[构建] 层 {layer_idx} | neutral_norm={n.norm().item():.4f} | "
            f"emotion_norm={e.norm().item():.4f} | diff_norm={ul.norm().item():.4f}"
        )

    if not unnormalized_vectors:
        raise RuntimeError("[构建] 没有可用层，无法构建 steering。")

    trans_map = load_transcription_map(dataset_dir)
    exclude_ids = neutral_pack.get("meta", {}).get("selected_file_ids", [])
    refs = build_search_references(
        dataset_dir=dataset_dir,
        neutral_subdir=neutral_label,
        transcription_map=trans_map,
        num_refs=cfg.search_samples,
        speaker_filter=parse_speaker_filter(speaker_filter),
        exclude_file_ids=exclude_ids,
        seed=cfg.sampling_seed if cfg.sampling_seed is not None else 1234,
        logger=logger,
    )
    ser_model = load_emotion2vec_model(cfg.emotion2vec_model, cfg.emotion2vec_hub, logger=logger)
    token_importance = evaluate_tokens_with_emotion2vec(
        runtime=runtime,
        ser_model=ser_model,
        layer_steering_vectors=unnormalized_vectors,
        selected_layers=active_layers,
        target_emotion=cfg.target_emotion,
        references=refs,
        nfe_step=cfg.nfe_step,
        cfg_strength=cfg.cfg_strength,
        sway_sampling_coef=cfg.sway_sampling_coef,
        sampling_seed=cfg.sampling_seed,
        logger=logger,
    )

    if not torch.isfinite(token_importance).all():
        raise RuntimeError("[构建] token 分数出现 NaN/Inf。")

    sorted_scores, sorted_indices = torch.sort(token_importance, descending=True)
    logger.info("[构建] 全部 token 排序分数如下（按降序）:")
    for rank, (idx, score) in enumerate(zip(sorted_indices.tolist(), sorted_scores.tolist()), start=1):
        logger.info(f"[TopK全排序] 排名={rank:04d} | token={idx:04d} | score={float(score):.8f}")

    num_tokens = int(token_importance.shape[0])
    top_k = int(cfg.top_k)
    if top_k <= 0 or top_k >= num_tokens:
        top_indices = sorted_indices
        top_values = sorted_scores
    else:
        top_indices = sorted_indices[:top_k]
        top_values = sorted_scores[:top_k]
    weights = F.softmax(top_values, dim=0)

    steering_steps: List[torch.Tensor] = []
    for ul in unnormalized_vectors:
        ul = ul.float()
        ul_norm = ul / (ul.norm(p=2) + 1e-8)

        layer_top_indices = top_indices.to(device=ul.device)
        layer_weights = weights.to(device=ul.device, dtype=ul.dtype)

        if layer_top_indices.numel() < ul.shape[1]:
            mask = torch.zeros(ul.shape[1], dtype=ul.dtype, device=ul.device)
            mask[layer_top_indices] = 1.0
            sl = ul_norm * mask.view(1, ul.shape[1], 1)
        else:
            sl = ul_norm

        wl = torch.zeros(ul.shape[1], dtype=ul.dtype, device=ul.device)
        wl[layer_top_indices] = layer_weights
        step_steer = torch.sum(sl * wl.view(1, ul.shape[1], 1), dim=1)  # [steps, d]
        step_steer = apply_step_aggregation_mode(step_steer, cfg.step_aggregation_mode)
        if cfg.post_agg_norm:
            step_steer = step_steer / (step_steer.norm(p=2) + 1e-8)
        if not torch.isfinite(step_steer).all():
            raise RuntimeError("[构建] step steering 出现 NaN/Inf。")
        steering_steps.append(step_steer.cpu())

    step_counts = {int(x.shape[0]) for x in steering_steps}
    if len(step_counts) != 1:
        raise RuntimeError(f"[构建] 各层 step 数不一致: {sorted(step_counts)}")
    expected_nfe_step = int(next(iter(step_counts)))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "layers": active_layers,
        "step_steering": steering_steps,
        "top_indices": top_indices.detach().cpu(),
        "top_values": top_values.detach().cpu(),
        "token_importance": token_importance.detach().cpu(),
        "token_sorted_indices": sorted_indices.detach().cpu(),
        "token_sorted_scores": sorted_scores.detach().cpu(),
        "top_k": int(top_k),
        "target_emotion": cfg.target_emotion,
        "use_emotion2vec": True,
        "source_neutral": str(neutral_residual_file),
        "source_emotion": str(emotion_residual_file),
        "expected_nfe_step": expected_nfe_step,
        "num_steps": expected_nfe_step,
        "meta": {
            "build_nfe_step": int(cfg.nfe_step),
            "build_cfg_strength": float(cfg.cfg_strength),
            "build_sway_sampling_coef": float(cfg.sway_sampling_coef),
            "build_sampling_seed": int(cfg.sampling_seed) if cfg.sampling_seed is not None else None,
            "step_aggregation_mode": cfg.step_aggregation_mode,
            "search_samples": int(cfg.search_samples),
        },
    }
    torch.save(bundle, output_file)

    logger.info(f"[构建] 已保存 steering bundle: {output_file}")
    logger.info(f"[构建] 生效层: {active_layers}")
    logger.info(f"[构建] top-k 数量: {len(top_indices)}")
    return bundle


def parse_stage_list(stages: str) -> List[str]:
    return [x.strip().lower() for x in stages.split(",") if x.strip()]


def choose_output_names(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = infer_dataset_tag(args.dataset_dir)
    emo_key = safe_name(args.emotion)
    prefix = f"{tag}_" if tag else ""

    neutral_name = args.neutral_residual_pt or f"{prefix}{args.neutral}_for_{emo_key}_residual.pt"
    emotion_name = args.emotion_residual_pt or f"{prefix}{emo_key}_residual.pt"
    steering_name = args.steering_pt or f"{prefix}{args.neutral}_to_{emo_key}_steering.pt"

    return output_dir / neutral_name, output_dir / emotion_name, output_dir / steering_name


def run_extract_stage(
    runtime: RuntimeHandles,
    args: argparse.Namespace,
    selected_layers: Sequence[int],
    neutral_residual_out: Path,
    emotion_residual_out: Path,
    logger: logging.Logger,
) -> None:
    cfg = ExtractionConfig(
        dataset_dir=args.dataset_dir,
        emotion=args.emotion,
        neutral=args.neutral,
        max_samples=args.max_samples,
        speaker_filter=args.speaker_filter,
        text_mode=args.text_mode,
        text_seed=args.text_seed,
        sampling_seed=args.seed,
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
    )

    speaker_filter = parse_speaker_filter(cfg.speaker_filter)
    neutral_files = collect_audio_files(
        dataset_dir=cfg.dataset_dir,
        emotion_subdir=cfg.neutral,
        speaker_filter=speaker_filter,
        max_samples=cfg.max_samples,
        sample_seed=cfg.sampling_seed,
    )
    emotion_seed = (cfg.sampling_seed + 100_003) if cfg.sampling_seed is not None else None
    emotion_files = collect_audio_files(
        dataset_dir=cfg.dataset_dir,
        emotion_subdir=cfg.emotion,
        speaker_filter=speaker_filter,
        max_samples=cfg.max_samples,
        sample_seed=emotion_seed,
    )

    if not neutral_files:
        raise RuntimeError(f"[提取] neutral 目录为空: {cfg.dataset_dir / cfg.neutral}")
    if not emotion_files:
        raise RuntimeError(f"[提取] emotion 目录为空: {cfg.dataset_dir / cfg.emotion}")

    logger.info(
        f"[提取] 样本统计 | neutral={len(neutral_files)} | emotion={len(emotion_files)} | "
        f"max_samples={cfg.max_samples}"
    )
    trans_map = load_transcription_map(cfg.dataset_dir)

    target_len_n = estimate_target_len_from_captured(
        runtime=runtime,
        emotion_subdir=cfg.neutral,
        selected_layers=selected_layers,
        audio_files=neutral_files,
        text_mode=cfg.text_mode,
        text_seed=cfg.text_seed,
        sampling_seed=cfg.sampling_seed,
        nfe_step=cfg.nfe_step,
        cfg_strength=cfg.cfg_strength,
        sway_sampling_coef=cfg.sway_sampling_coef,
        transcription_map=trans_map,
        logger=logger,
    )
    target_len_e = estimate_target_len_from_captured(
        runtime=runtime,
        emotion_subdir=cfg.emotion,
        selected_layers=selected_layers,
        audio_files=emotion_files,
        text_mode=cfg.text_mode,
        text_seed=cfg.text_seed,
        sampling_seed=cfg.sampling_seed,
        nfe_step=cfg.nfe_step,
        cfg_strength=cfg.cfg_strength,
        sway_sampling_coef=cfg.sway_sampling_coef,
        transcription_map=trans_map,
        logger=logger,
    )
    target_len = max(1, int(round((target_len_n + target_len_e) / 2)))
    target_len_source = "captured_first_residual_avg"
    logger.info(
        f"[提取] 共享目标 token 长度: {target_len} "
        f"(neutral={target_len_n}, emotion={target_len_e})"
    )

    neutral_pack = extract_mean_activation(
        runtime=runtime,
        dataset_dir=cfg.dataset_dir,
        emotion_subdir=cfg.neutral,
        selected_layers=selected_layers,
        audio_files=neutral_files,
        target_len=target_len,
        target_len_source=target_len_source,
        cfg=cfg,
        transcription_map=trans_map,
        logger=logger,
    )
    neutral_pack["neutral"] = cfg.neutral

    emotion_pack = extract_mean_activation(
        runtime=runtime,
        dataset_dir=cfg.dataset_dir,
        emotion_subdir=cfg.emotion,
        selected_layers=selected_layers,
        audio_files=emotion_files,
        target_len=target_len,
        target_len_source=target_len_source,
        cfg=cfg,
        transcription_map=trans_map,
        logger=logger,
    )
    emotion_pack["neutral"] = cfg.neutral

    neutral_residual_out.parent.mkdir(parents=True, exist_ok=True)
    emotion_residual_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(neutral_pack, neutral_residual_out)
    torch.save(emotion_pack, emotion_residual_out)
    logger.info(f"[提取] 已保存 neutral 残差: {neutral_residual_out}")
    logger.info(f"[提取] 已保存 emotion 残差: {emotion_residual_out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EmoSteer 论文 3.2/3.3 单脚本实现（数据读取 -> 第一残差提取 -> token 打分 -> top-k）"
    )
    parser.add_argument("--stages", type=str, default="extract,build", help="要执行的阶段，逗号分隔: extract,build,score,convert")

    parser.add_argument("--dataset_dir", type=Path, required=True, help="数据集根目录，内部应包含 neutral/emotion 子目录")
    parser.add_argument("--emotion", type=str, required=True, help="目标情绪目录名，例如 sad")
    parser.add_argument("--neutral", type=str, default="neutral", help="中性目录名")
    parser.add_argument("--speaker_filter", type=str, default=None, help="说话人筛选，如 0010-0015 或 0010,0011")
    parser.add_argument("--max_samples", type=int, default=250, help="每个情绪用于残差提取的最大样本数")
    parser.add_argument("--search_samples", type=int, default=10, help="emotion2vec token 搜索的参考样本数")

    parser.add_argument("--layers", type=str, default="paper", help="层选择: paper/all/逗号分隔索引")
    parser.add_argument("--top_k", type=int, default=200, help="top-k token 数量")
    parser.add_argument("--seed", type=int, default=None, help="全流程随机种子")
    parser.add_argument("--text_mode", type=str, choices=["random_pool", "ref_text"], default="random_pool")
    parser.add_argument("--text_seed", type=int, default=1234)

    parser.add_argument("--nfe_step", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--sway_sampling_coef", type=float, default=-1.0)
    parser.add_argument("--step_aggregation_mode", type=str, choices=["per_step", "mean_repeat"], default="per_step")
    parser.add_argument("--post_agg_norm", action="store_true")

    parser.add_argument("--model_name", type=str, default="F5TTS_v1_Base")
    parser.add_argument("--vocoder_name", type=str, default="vocos")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model_cfg", type=str, default=None)
    parser.add_argument("--ckpt_file", type=str, default=None)
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--load_vocoder_from_local", action="store_true")
    parser.add_argument("--f5_repo_dir", type=Path, default=None, help="可选：f5_tts 仓库根目录（脚本会自动加 /src 到 sys.path）")

    parser.add_argument("--emotion2vec_model", type=str, default="iic/emotion2vec_plus_large")
    parser.add_argument("--emotion2vec_hub", type=str, default="ms")

    parser.add_argument("--output_dir", type=Path, default=Path("./outputs_3233"))
    parser.add_argument("--neutral_residual_pt", type=str, default="", help="neutral 残差输出文件名（.pt）")
    parser.add_argument("--emotion_residual_pt", type=str, default="", help="emotion 残差输出文件名（.pt）")
    parser.add_argument("--steering_pt", type=str, default="", help="steering bundle 输出文件名（.pt）")
    parser.add_argument("--log_file", type=Path, default=None, help="日志文件路径")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.log_file is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.log_file = args.output_dir / "emosteer_3233.log"
    logger = setup_logger(args.log_file)

    logger.info("========== EmoSteer 3.2/3.3 单脚本流程开始 ==========")
    logger.info(
        f"[参数] emotion={args.emotion} | neutral={args.neutral} | stages={args.stages} | "
        f"top_k={args.top_k} | max_samples={args.max_samples} | search_samples={args.search_samples}"
    )
    set_global_seed(args.seed)

    runtime_cfg = RuntimeConfig(
        model_name=args.model_name,
        vocoder_name=args.vocoder_name,
        device=args.device,
        model_cfg=args.model_cfg,
        ckpt_file=args.ckpt_file,
        vocab_file=args.vocab_file,
        load_vocoder_from_local=bool(args.load_vocoder_from_local),
        f5_repo_dir=args.f5_repo_dir,
    )
    runtime = load_runtime(runtime_cfg, logger=logger)
    patch_ditblock_forward_if_needed(runtime.model, logger=logger)

    total_layers = len(runtime.model.transformer.transformer_blocks)
    selected_layers = parse_layers(args.layers, total_layers=total_layers, model_name=args.model_name)
    if not selected_layers:
        raise RuntimeError("层选择结果为空，请检查 --layers 参数。")
    logger.info(f"[参数] 选中层: {selected_layers}")

    neutral_residual_out, emotion_residual_out, steering_out = choose_output_names(args)
    logger.info(
        f"[输出] neutral_residual={neutral_residual_out.name} | "
        f"emotion_residual={emotion_residual_out.name} | steering={steering_out.name}"
    )

    stages = parse_stage_list(args.stages)
    valid_stages = {"extract", "build", "score", "convert"}
    illegal = [x for x in stages if x not in valid_stages]
    if illegal:
        raise ValueError(f"非法 stages: {illegal}，仅支持 {sorted(valid_stages)}")

    if "extract" in stages:
        run_extract_stage(
            runtime=runtime,
            args=args,
            selected_layers=selected_layers,
            neutral_residual_out=neutral_residual_out,
            emotion_residual_out=emotion_residual_out,
            logger=logger,
        )

    if "build" in stages or "score" in stages:
        if not neutral_residual_out.exists() or not emotion_residual_out.exists():
            raise FileNotFoundError(
                "[构建] 缺少残差文件。请先运行 extract 阶段，或通过 --output_dir/文件名 指向已有残差。"
            )
        bcfg = BuildConfig(
            top_k=args.top_k,
            target_emotion=args.emotion,
            search_samples=args.search_samples,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            sampling_seed=args.seed,
            emotion2vec_model=args.emotion2vec_model,
            emotion2vec_hub=args.emotion2vec_hub,
            step_aggregation_mode=args.step_aggregation_mode,
            post_agg_norm=bool(args.post_agg_norm),
        )
        build_steering_bundle(
            runtime=runtime,
            neutral_residual_file=neutral_residual_out,
            emotion_residual_file=emotion_residual_out,
            output_file=steering_out,
            cfg=bcfg,
            dataset_dir=args.dataset_dir,
            neutral_label=args.neutral,
            speaker_filter=args.speaker_filter,
            logger=logger,
        )

    if "convert" in stages:
        logger.info("[Convert] 当前脚本暂未实现 convert 阶段，已预留 stage 参数。")

    logger.info("========== 全流程结束 ==========")
    logger.info(f"[日志] 已写入: {args.log_file}")


if __name__ == "__main__":
    main()

