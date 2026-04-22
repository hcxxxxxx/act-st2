#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import contextlib
import io
import importlib
import inspect
import json
import logging
import random
import re
import shutil
import subprocess
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

AUDIO_SUFFIXES: Set[str] = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


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
    auto_bootstrap_f5: bool
    f5_vendor_dir: Optional[Path]
    f5_git_url: str
    f5_git_ref: str


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
    min_ref_tokens: int
    target_len_mode: str
    debug_verbose: bool
    min_ref_text_en_words: int
    min_ref_text_zh_chars: int
    prune_bad_samples: bool


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
    min_search_ref_tokens: int
    step_aggregation_mode: str
    post_agg_norm: bool
    debug_verbose: bool
    min_ref_text_en_words: int
    min_ref_text_zh_chars: int


@dataclass
class ConvertConfig:
    steering_bundle: Optional[Path]
    ref_audio: Optional[Path]
    ref_text: str
    gen_text: str
    alpha: float
    output_wav: Optional[Path]
    nfe_step_override: int
    cfg_strength: float
    sway_sampling_coef: float
    sampling_seed: Optional[int]


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
    if name in {"datasets_merged_intersection", "merged_intersection"} or "merged_intersection" in name:
        return "merged_intersection"
    if "emilia" in name:
        return "emo_emilia"
    return safe_name(name)


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def count_cjk_chars(text: str) -> int:
    return sum(1 for ch in str(text) if "\u4e00" <= ch <= "\u9fff")


def count_en_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(text)))


def is_ref_text_too_short(text: str, min_en_words: int, min_zh_chars: int) -> bool:
    txt = str(text or "").strip()
    if not txt:
        return True
    if contains_cjk(txt):
        return count_cjk_chars(txt) < int(min_zh_chars)
    return count_en_words(txt) < int(min_en_words)


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


def _normalize_speaker_id(speaker: str) -> str:
    s = str(speaker).strip()
    if not s:
        return ""
    if s.isdigit():
        return str(int(s))
    return s.lower()


def _speaker_match(speaker: str, speaker_filter: Set[str]) -> bool:
    if not speaker_filter:
        return True
    raw = str(speaker).strip()
    if raw in speaker_filter:
        return True
    norm_raw = _normalize_speaker_id(raw)
    if not norm_raw:
        return False
    for x in speaker_filter:
        if norm_raw == _normalize_speaker_id(x):
            return True
    return False


def _iter_audio_files(base: Path) -> List[Path]:
    if not base.exists():
        return []
    files: List[Path] = []
    for p in sorted(base.rglob("*")):
        if p.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        if p.is_file() or p.is_symlink():
            files.append(p)
    return files


def _load_manifest_speaker_map(dataset_dir: Path, emotion_subdir: str) -> Dict[str, str]:
    """
    merged 数据集结构下，从 manifests/<emotion>.jsonl 读取 uid->speaker 映射。
    """
    out: Dict[str, str] = {}
    manifest = dataset_dir / "manifests" / f"{emotion_subdir}.jsonl"
    if not manifest.exists():
        return out
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            uid = str(row.get("uid", "")).strip()
            if not uid:
                continue
            speaker = str(row.get("speaker", "")).strip()
            out[uid] = speaker
    return out


def balanced_sample_by_speaker(
    files: Sequence[Path],
    max_samples: int,
    seed: Optional[int],
    speaker_map: Optional[Dict[str, str]] = None,
) -> List[Path]:
    if max_samples <= 0 or len(files) <= max_samples:
        return list(files)
    rng = random.Random(seed)
    by_speaker: Dict[str, List[Path]] = {}
    for wav in files:
        sid = ""
        if speaker_map is not None:
            sid = str(speaker_map.get(wav.stem, "")).strip()
        if not sid:
            sid = speaker_id_from_file(wav)
        by_speaker.setdefault(sid, []).append(wav)

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
    merged_base = dataset_dir / "audio" / emotion_subdir
    legacy_base = dataset_dir / emotion_subdir
    use_merged = merged_base.exists()
    base = merged_base if use_merged else legacy_base

    files = _iter_audio_files(base)
    speaker_map: Optional[Dict[str, str]] = None
    if use_merged:
        speaker_map = _load_manifest_speaker_map(dataset_dir, emotion_subdir)

    if speaker_filter:
        if speaker_map:
            files = [x for x in files if _speaker_match(speaker_map.get(x.stem, ""), speaker_filter)]
        else:
            files = [x for x in files if _speaker_match(speaker_id_from_file(x), speaker_filter)]
    if max_samples > 0:
        files = balanced_sample_by_speaker(
            files,
            max_samples=max_samples,
            seed=sample_seed,
            speaker_map=speaker_map,
        )
    return files


def load_transcription_map(dataset_dir: Path) -> Dict[str, str]:
    # 新结构优先：<dataset_dir>/subtitles/subtitles.jsonl
    subtitles_jsonl = dataset_dir / "subtitles" / "subtitles.jsonl"
    mapping: Dict[str, str] = {}
    if subtitles_jsonl.exists():
        with subtitles_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                text = str(row.get("text", "")).strip()
                if not text:
                    continue

                uid = str(row.get("uid", "")).strip()
                if uid and uid not in mapping:
                    mapping[uid] = text

                audio = str(row.get("audio", row.get("src_audio", ""))).strip()
                if audio:
                    stem = Path(audio).stem
                    if stem and stem not in mapping:
                        mapping[stem] = text

        if mapping:
            return mapping

    # 旧结构兜底：<dataset_dir>/transcription/*.txt
    tdir = dataset_dir / "transcription"
    if not tdir.exists():
        return mapping
    for txt in sorted(tdir.glob("*.txt")):
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def prune_subtitles_jsonl(
    dataset_dir: Path,
    removed_file_ids: Set[str],
    logger: logging.Logger,
) -> int:
    subtitles_jsonl = dataset_dir / "subtitles" / "subtitles.jsonl"
    if not subtitles_jsonl.exists() or not removed_file_ids:
        return 0

    kept_lines: List[str] = []
    removed_rows = 0
    with subtitles_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception:
                kept_lines.append(raw)
                continue

            uid = str(row.get("uid", "")).strip()
            audio = str(row.get("audio", row.get("src_audio", ""))).strip()
            stem = Path(audio).stem if audio else ""
            if uid in removed_file_ids or stem in removed_file_ids:
                removed_rows += 1
                continue
            kept_lines.append(raw)

    with subtitles_jsonl.open("w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    logger.info(f"[清理] subtitles.jsonl 已更新 | 删除条目={removed_rows}")
    return removed_rows


def prune_manifest_jsonl(
    dataset_dir: Path,
    emotion_subdir: str,
    removed_file_ids: Set[str],
    logger: logging.Logger,
) -> int:
    manifest = dataset_dir / "manifests" / f"{emotion_subdir}.jsonl"
    if not manifest.exists() or not removed_file_ids:
        return 0

    kept_lines: List[str] = []
    removed_rows = 0
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception:
                kept_lines.append(raw)
                continue
            uid = str(row.get("uid", "")).strip()
            if uid in removed_file_ids:
                removed_rows += 1
                continue
            kept_lines.append(raw)

    with manifest.open("w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    logger.info(f"[清理] manifest({emotion_subdir}) 已更新 | 删除条目={removed_rows}")
    return removed_rows


def prune_bad_samples_from_dataset(
    dataset_dir: Path,
    emotion_subdir: str,
    transcription_map: Dict[str, str],
    min_ref_tokens: int,
    min_ref_text_en_words: int,
    min_ref_text_zh_chars: int,
    logger: logging.Logger,
) -> Dict[str, int]:
    files = collect_audio_files(
        dataset_dir=dataset_dir,
        emotion_subdir=emotion_subdir,
        speaker_filter=None,
        max_samples=0,
        sample_seed=None,
    )
    if not files:
        return {"total": 0, "removed": 0, "removed_short_token": 0, "removed_short_text": 0}

    removed_ids: Set[str] = set()
    removed_short_token = 0
    removed_short_text = 0
    removed_total = 0

    for wav in files:
        file_id = wav.stem
        ref_text = transcription_map.get(file_id, "")
        ref_tokens = estimate_ref_audio_token_len_compatible(str(wav), ref_text=ref_text)
        short_token = ref_tokens < int(min_ref_tokens)
        short_text = is_ref_text_too_short(
            ref_text,
            min_en_words=int(min_ref_text_en_words),
            min_zh_chars=int(min_ref_text_zh_chars),
        )
        if not (short_token or short_text):
            continue
        try:
            wav.unlink()
            removed_total += 1
            removed_ids.add(file_id)
            if short_token:
                removed_short_token += 1
            if short_text:
                removed_short_text += 1
        except Exception as e:
            logger.warning(f"[清理] 删除失败，跳过: {wav} | err={e}")

    if removed_ids:
        prune_manifest_jsonl(dataset_dir, emotion_subdir, removed_ids, logger)
        prune_subtitles_jsonl(dataset_dir, removed_ids, logger)

    logger.info(
        f"[清理] emotion={emotion_subdir} | total={len(files)} | removed={removed_total} | "
        f"short_token={removed_short_token} | short_text={removed_short_text}"
    )
    return {
        "total": len(files),
        "removed": removed_total,
        "removed_short_token": removed_short_token,
        "removed_short_text": removed_short_text,
    }


def build_text_pools(
    trans_map: Dict[str, str],
    seed: int,
    min_ref_text_en_words: int,
    min_ref_text_zh_chars: int,
) -> Dict[str, List[str]]:
    pools = {"all": [], "en": [], "zh": []}
    seen: Set[str] = set()
    for _, text in sorted(trans_map.items()):
        text = text.strip()
        if not text or text in seen:
            continue
        if is_ref_text_too_short(
            text,
            min_en_words=int(min_ref_text_en_words),
            min_zh_chars=int(min_ref_text_zh_chars),
        ):
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


def sanitize_gen_text_for_single_batch(
    gen_text: str,
    ref_text: str = "",
    fallback_text: str = "This is a sample text for activation extraction",
    max_bytes: int = 80,
) -> str:
    """
    规整生成文本，尽量确保 F5 infer_process 只产生 1 个文本 batch：
    - 取首行、首句；
    - 去掉会触发 chunk_text 分句的标点；
    - 控制长度，避免极端时长估计。
    """
    txt = str(gen_text or "").strip()
    if not txt:
        txt = str(ref_text or "").strip()
    if not txt:
        txt = fallback_text

    txt = txt.splitlines()[0].strip()
    txt = re.split(r"[。！？!?;；:：]+", txt, maxsplit=1)[0].strip()
    txt = re.sub(r"[,，、.。]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    if len(txt.encode("utf-8")) > int(max_bytes):
        cut = txt
        while cut and len(cut.encode("utf-8")) > int(max_bytes):
            cut = cut[:-1]
        txt = cut.strip()

    if not txt:
        txt = fallback_text
    return txt


def short_text_for_log(text: str, max_len: int = 120) -> str:
    txt = str(text or "").replace("\n", " ").strip()
    if len(txt) <= int(max_len):
        return txt
    return txt[: int(max_len)] + "..."


def log_infer_call_detail(
    logger: logging.Logger,
    stage: str,
    emotion: str,
    file_id: str,
    ref_audio: str,
    ref_tokens: int,
    ref_text: str,
    gen_text: str,
    nfe_step: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    seed: Optional[int],
    extra: str = "",
) -> None:
    suffix = f" | {extra}" if str(extra).strip() else ""
    logger.info(
        f"[推理调用-细节] stage={stage} | emotion={emotion} | file_id={file_id} | "
        f"ref_audio={ref_audio} | ref_tokens={int(ref_tokens)} | "
        f"ref_text='{short_text_for_log(ref_text)}' | gen_text='{short_text_for_log(gen_text)}' | "
        f"nfe_step={int(nfe_step)} | cfg_strength={float(cfg_strength):.4f} | "
        f"sway_sampling_coef={float(sway_sampling_coef):.4f} | seed={seed}{suffix}"
    )


def select_search_gen_text_by_lang(lang: str) -> str:
    # token 搜索阶段固定使用中长句，避免随机抽到单词级文本导致评分退化。
    if str(lang).lower() == "zh":
        return "今天下午三点，研究小组在会议室汇报了实验结果，并记录了下一步安排。"
    return "At three o'clock this afternoon, the research team presented the experiment results and recorded the next steps in the meeting room."


def _is_f5_tts_importable() -> bool:
    try:
        importlib.import_module("f5_tts")
        return True
    except ModuleNotFoundError:
        return False


def ensure_f5_tts_importable(config: RuntimeConfig, logger: logging.Logger) -> None:
    # 1) 用户显式指定了仓库目录：仅使用该目录
    if config.f5_repo_dir is not None:
        src_dir = config.f5_repo_dir / "src"
        if not src_dir.exists():
            raise FileNotFoundError(f"未找到 f5_tts 源码目录: {src_dir}")
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        if not _is_f5_tts_importable():
            raise ModuleNotFoundError(f"已添加 {src_dir} 到 sys.path，但仍无法导入 f5_tts。")
        logger.info(f"[运行时] 使用 --f5_repo_dir 指定的本地 f5_tts: {config.f5_repo_dir}")
        return

    # 2) 已安装在当前环境（site-packages）
    if _is_f5_tts_importable():
        logger.info("[运行时] 检测到当前环境已安装 f5_tts，直接使用。")
        return

    # 3) 环境未安装 -> 自动在当前项目目录内自举
    if not config.auto_bootstrap_f5:
        raise ModuleNotFoundError(
            "当前环境未安装 f5_tts，且已关闭自动自举。"
            "可用 --f5_repo_dir 指定源码目录，或启用自动自举。"
        )

    vendor_dir = config.f5_vendor_dir
    if vendor_dir is None:
        vendor_dir = Path(__file__).resolve().parent / "_vendor" / "F5-TTS"
    vendor_dir = vendor_dir.resolve()
    src_dir = vendor_dir / "src"

    if not (src_dir / "f5_tts").exists():
        vendor_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = ["git", "clone", "--depth", "1"]
        if config.f5_git_ref:
            clone_cmd.extend(["--branch", config.f5_git_ref])
        clone_cmd.extend([config.f5_git_url, str(vendor_dir)])
        logger.info(f"[运行时] 未找到 f5_tts，开始自举到本目录: {vendor_dir}")
        try:
            subprocess.run(clone_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "").strip()
            raise RuntimeError(
                "自动拉取 F5-TTS 失败。"
                f"\n命令: {' '.join(clone_cmd)}"
                f"\n错误: {err or '无详细错误输出'}"
            ) from e

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if not _is_f5_tts_importable():
        raise ModuleNotFoundError(
            f"已自举到 {vendor_dir}，但仍无法导入 f5_tts。"
            "请检查该目录完整性，或手动安装依赖后重试。"
        )
    logger.info(f"[运行时] 已使用本项目目录内的 f5_tts: {vendor_dir}")


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
    ensure_f5_tts_importable(config, logger)

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
    key = str(label or "").strip().lower()
    if key in EMOTION_ALIAS:
        return EMOTION_ALIAS[key]
    raise ValueError(
        f"不支持的情绪标签: '{label}'。支持: {sorted(set(EMOTION_ALIAS.keys()))}"
    )


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
            # 由上层 DiT.forward 标记：当前是否无条件分支（drop_audio_cond=True）。
            is_uncond_branch = bool(getattr(self, "_emosteer_drop_audio_cond", False))
            ref_audio_len = getattr(self, "current_ref_audio_len", None)
            if ref_audio_len is None:
                ref_audio_len = x.shape[1]
            ref_audio_len = max(0, min(int(ref_audio_len), int(x.shape[1])))

            cond_batch = getattr(self, "cfg_batch_size", None)
            if cond_batch is None:
                cond_batch = x.shape[0]
            cond_batch = max(1, min(int(cond_batch), int(x.shape[0])))

            step_steering = getattr(self, "step_steering", None)
            # 关键修复：仅在 cond 分支注入，避免与 uncond 分支在 CFG 中相互抵消。
            if is_uncond_branch:
                step_steering = None
            if step_steering is not None and ref_audio_len > 0:
                if not torch.is_tensor(step_steering):
                    raise RuntimeError("step_steering 不是张量，无法执行 token 注入。")
                if step_steering.ndim == 1:
                    act = step_steering
                elif step_steering.ndim == 2:
                    # 关键修复：step 计数仅按 cond 分支递增。
                    call_counter = int(getattr(self, "_step_call_counter_cond", 0))
                    raw_step_idx = getattr(self, "current_step_idx", None)
                    if raw_step_idx is None:
                        raw_step_idx = call_counter
                    step_idx = max(0, min(int(raw_step_idx), int(step_steering.shape[0]) - 1))
                    act = step_steering[step_idx]
                    self._step_call_counter_cond = call_counter + 1
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

                # 参考附录实现：注入前先对 act 单位化，再按 Eq.2/Eq.8 的 f_r 做范数回缩。
                orig_norm = x.norm(p=2, dim=(1, 2), keepdim=True)
                x = x + alpha * steer
                new_norm = x.norm(p=2, dim=(1, 2), keepdim=True) + 1e-8
                x = x * (orig_norm / new_norm)

            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
            attn_output = self.attn(x=norm, mask=mask, rope=rope)

            x_res1 = x + gate_msa.unsqueeze(1) * attn_output

            # 关键修复：只捕获 cond 分支第一残差，避免 uncond 污染均值残差。
            if bool(getattr(self, "save_residual", False)) and (not is_uncond_branch):
                self.first_residual = x_res1.detach()
                if ref_audio_len > 0:
                    captured = x_res1[0, :ref_audio_len, :].detach().cpu()
                    total_steps = getattr(self, "current_total_steps", None)
                    capture_counter = int(getattr(self, "_capture_call_counter_cond", 0))
                    cap_idx = getattr(self, "current_step_idx", None)
                    if cap_idx is None:
                        cap_idx = capture_counter
                    self._capture_call_counter_cond = capture_counter + 1

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
        block._step_call_counter_cond = 0
        block._capture_call_counter_cond = 0
        block._emosteer_drop_audio_cond = False
        patched += 1

    # 关键修复：patch DiT.forward，把 drop_audio_cond 状态传给各个 block。
    transformer = getattr(model, "transformer", None)
    if transformer is not None and not bool(getattr(transformer, "_emosteer_drop_hooked", False)):
        orig_forward = transformer.forward
        try:
            sig = inspect.signature(orig_forward)
        except Exception:
            sig = None

        def patched_transformer_forward(self, *args, **kwargs):
            drop_audio_cond = False
            if sig is not None:
                try:
                    bound = sig.bind_partial(*args, **kwargs)
                    if "drop_audio_cond" in bound.arguments:
                        drop_audio_cond = bool(bound.arguments["drop_audio_cond"])
                    else:
                        drop_audio_cond = bool(kwargs.get("drop_audio_cond", False))
                except Exception:
                    drop_audio_cond = bool(kwargs.get("drop_audio_cond", False))
            else:
                drop_audio_cond = bool(kwargs.get("drop_audio_cond", False))

            for blk in self.transformer_blocks:
                blk._emosteer_drop_audio_cond = drop_audio_cond
            try:
                return orig_forward(*args, **kwargs)
            finally:
                for blk in self.transformer_blocks:
                    blk._emosteer_drop_audio_cond = False

        transformer.forward = MethodType(patched_transformer_forward, transformer)
        transformer._emosteer_drop_hooked = True
        logger.info("[Hook] 已 patch DiT.forward：按 drop_audio_cond 区分 cond/uncond 分支。")

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
        block._step_call_counter_cond = 0
        block._capture_call_counter_cond = 0
        block._emosteer_drop_audio_cond = False


def enable_residual_capture(model: torch.nn.Module, selected_layers: Sequence[int], nfe_step: int) -> None:
    reset_all_blocks(model)
    for layer_idx in selected_layers:
        block = model.transformer.transformer_blocks[layer_idx]
        block.save_residual = True
        block.step_residual_tokens = [None] * int(nfe_step)
        block.current_total_steps = int(nfe_step)
        block._capture_call_counter = 0
        block._capture_call_counter_cond = 0


def disable_residual_capture(model: torch.nn.Module) -> None:
    for block in model.transformer.transformer_blocks:
        block.save_residual = False


def clear_token_steering(model: torch.nn.Module, selected_layers: Sequence[int]) -> None:
    for layer_idx in selected_layers:
        block = model.transformer.transformer_blocks[layer_idx]
        block.step_steering = None
        block.alpha = 0.0
        block._step_call_counter = 0
        block._step_call_counter_cond = 0


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
        block._step_call_counter_cond = 0


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


def estimate_ref_audio_token_len_compatible(wav_path: str, ref_text: str = "", hop_length: int = 256) -> int:
    """
    与主流程一致的 token 长度估计：优先走 preprocess_ref_audio_text 后再估计。
    若预处理失败，回退到原始音频估计。
    """
    try:
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        ref_audio_path, _ = preprocess_ref_audio_text(str(wav_path), str(ref_text or ""))
        return estimate_ref_audio_token_len(str(ref_audio_path), hop_length=hop_length)
    except Exception:
        return estimate_ref_audio_token_len(str(wav_path), hop_length=hop_length)


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


_INFER_DROPPED_ARGS_WARNED: Set[str] = set()


def call_infer_process_compat(
    infer_process_fn: Any,
    call_kwargs: Dict[str, Any],
    seed: Optional[int],
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    兼容不同版本 f5_tts.infer.utils_infer.infer_process 的参数签名：
    - 若函数支持 seed，则直接传入；
    - 若不支持 seed，则调用前手动设随机种子；
    - 仅传入目标函数签名里存在的参数，避免 unexpected keyword 报错。
    """
    sig = inspect.signature(infer_process_fn)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs = dict(call_kwargs)
    if seed is not None:
        if has_var_kw or "seed" in params:
            kwargs["seed"] = int(seed)
        else:
            set_global_seed(int(seed))

    if not has_var_kw:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        dropped = sorted(k for k in kwargs.keys() if k not in params)
        if dropped and logger is not None:
            key = ",".join(dropped)
            if key not in _INFER_DROPPED_ARGS_WARNED:
                _INFER_DROPPED_ARGS_WARNED.add(key)
                logger.warning(f"[兼容] 当前 infer_process 不支持参数 {dropped}，已自动忽略。")
    else:
        filtered_kwargs = kwargs

    return infer_process_fn(**filtered_kwargs)


def call_infer_process_with_retry(
    infer_process_fn: Any,
    call_kwargs: Dict[str, Any],
    seed: Optional[int],
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    调用 infer_process，并在常见的长度不一致错误上做一次短文本重试。
    """
    try:
        return call_infer_process_compat(
            infer_process_fn=infer_process_fn,
            call_kwargs=call_kwargs,
            seed=seed,
            logger=logger,
        )
    except RuntimeError as e:
        msg = str(e)
        if "Sizes of tensors must match" not in msg:
            raise

        retry_kwargs = dict(call_kwargs)
        fallback_text = "This is a sample text for activation extraction"
        retry_kwargs["gen_text"] = sanitize_gen_text_for_single_batch(
            gen_text=fallback_text,
            ref_text=str(call_kwargs.get("ref_text", "")),
            fallback_text=fallback_text,
            max_bytes=48,
        )
        if logger is not None:
            logger.warning("[兼容] infer_process 出现长度不一致，已使用短文本重试一次。")
        return call_infer_process_compat(
            infer_process_fn=infer_process_fn,
            call_kwargs=retry_kwargs,
            seed=seed,
            logger=logger,
        )


def normalize_infer_output(output: Any) -> Tuple[Any, int, Any]:
    """
    统一 infer_process 返回值格式，兼容 tuple/dict/单值。
    返回: (wav, sample_rate, meta)
    """
    if isinstance(output, tuple):
        if len(output) >= 2:
            meta = output[2] if len(output) >= 3 else None
            return output[0], int(output[1]), meta
        if len(output) == 1:
            return output[0], 24000, None
    if isinstance(output, dict):
        wav = output.get("wav", output.get("audio", output.get("audio_np", None)))
        sr = int(output.get("sample_rate", output.get("sr", 24000)))
        return wav, sr, output
    return output, 24000, None


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
        "min_ref_tokens": int(cfg.min_ref_tokens),
        "debug_verbose": bool(cfg.debug_verbose),
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
    min_ref_tokens: int,
    min_ref_text_en_words: int,
    min_ref_text_zh_chars: int,
    fast_ref_only: bool,
    debug_verbose: bool,
    transcription_map: Dict[str, str],
    logger: logging.Logger,
) -> int:
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    if not audio_files:
        raise RuntimeError(f"[提取] 情绪目录 '{emotion_subdir}' 没有 wav 文件。")

    text_pools = build_text_pools(
        transcription_map,
        seed=text_seed,
        min_ref_text_en_words=int(min_ref_text_en_words),
        min_ref_text_zh_chars=int(min_ref_text_zh_chars),
    )
    if not fast_ref_only:
        enable_residual_capture(runtime.model, selected_layers, nfe_step=nfe_step)

    captured_lens: List[int] = []
    skipped_short = 0
    skipped_no_capture = 0
    mode_name = "ref_audio(无inference)" if fast_ref_only else "captured(含inference)"
    logger.info(
        f"[提取] 估计目标 token 长度 | 情绪={emotion_subdir} | 样本数={len(audio_files)} | 模式={mode_name}"
    )

    try:
        for i, wav_path in enumerate(audio_files, start=1):
            file_id = wav_path.stem
            ref_text = transcription_map.get(file_id, "")
            if text_mode == "random_pool":
                gen_text = select_generation_text(text_pools, i - 1, ref_text)
            else:
                gen_text = ref_text or "This is a sample text for activation extraction"
            gen_text = sanitize_gen_text_for_single_batch(gen_text=gen_text, ref_text=ref_text)

            if not fast_ref_only:
                for layer_idx in selected_layers:
                    block = runtime.model.transformer.transformer_blocks[layer_idx]
                    block.step_residual_tokens = [None] * int(nfe_step)
                    block._capture_call_counter = 0

            ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(str(wav_path), ref_text)
            ref_audio_len = estimate_ref_audio_token_len(str(ref_audio_processed))
            if debug_verbose:
                logger.info(
                    f"[提取-细节] 长度估计样本 | idx={i} | file={wav_path.name} | "
                    f"ref_tokens={ref_audio_len} | gen_text='{gen_text[:80]}'"
                )
            if ref_audio_len < int(min_ref_tokens):
                skipped_short += 1
                if skipped_short <= 20:
                    logger.warning(
                        f"[提取] 参考token过短，跳过样本 {wav_path.name} | "
                        f"ref_tokens={ref_audio_len} < min_ref_tokens={int(min_ref_tokens)}"
                    )
                continue
            if fast_ref_only:
                captured_lens.append(int(ref_audio_len))
                if i % 20 == 0 or i == len(audio_files):
                    logger.info(f"[提取] 目标长度估计进度: {i}/{len(audio_files)}")
                continue

            set_runtime_context_for_all_blocks(runtime.model, nfe_step=nfe_step, ref_audio_len=ref_audio_len, cfg_batch_size=1)
            if debug_verbose:
                infer_seed = (sampling_seed + i - 1) if sampling_seed is not None else None
                log_infer_call_detail(
                    logger=logger,
                    stage="extract.estimate_target_len",
                    emotion=emotion_subdir,
                    file_id=file_id,
                    ref_audio=str(ref_audio_processed),
                    ref_tokens=int(ref_audio_len),
                    ref_text=ref_text_processed,
                    gen_text=gen_text,
                    nfe_step=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    seed=infer_seed,
                )

            call_infer_process_with_retry(
                infer_process_fn=infer_process,
                call_kwargs={
                    "ref_audio": ref_audio_processed,
                    "ref_text": ref_text_processed,
                    "gen_text": gen_text,
                    "model_obj": runtime.model,
                    "vocoder": runtime.vocoder,
                    "mel_spec_type": runtime.vocoder_name,
                    "nfe_step": nfe_step,
                    "cfg_strength": cfg_strength,
                    "sway_sampling_coef": sway_sampling_coef,
                    "device": runtime.device,
                },
                seed=(sampling_seed + i - 1) if sampling_seed is not None else None,
                logger=logger,
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
                if debug_verbose:
                    logger.info(
                        f"[提取-细节] 捕获长度 | idx={i} | file={wav_path.name} | captured_tokens={sample_len}"
                    )
            else:
                skipped_no_capture += 1
                if debug_verbose:
                    logger.warning(
                        f"[提取-细节] 未捕获到残差长度 | idx={i} | file={wav_path.name}"
                    )
            if i % 20 == 0 or i == len(audio_files):
                logger.info(f"[提取] 目标长度估计进度: {i}/{len(audio_files)}")
    finally:
        if not fast_ref_only:
            disable_residual_capture(runtime.model)

    if not captured_lens:
        raise RuntimeError(
            f"[提取] 未能估计目标 token 长度: {emotion_subdir} | "
            f"skip_short={skipped_short} | skip_no_capture={skipped_no_capture}"
        )

    logger.info(
        f"[提取] 长度估计过滤统计 | skip_short={skipped_short} | skip_no_capture={skipped_no_capture}"
    )

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

    text_pools = build_text_pools(
        transcription_map,
        seed=cfg.text_seed,
        min_ref_text_en_words=int(cfg.min_ref_text_en_words),
        min_ref_text_zh_chars=int(cfg.min_ref_text_zh_chars),
    )
    enable_residual_capture(runtime.model, selected_layers, nfe_step=cfg.nfe_step)

    layer_sums: List[Optional[torch.Tensor]] = [None for _ in selected_layers]
    layer_counts: List[int] = [0 for _ in selected_layers]
    skipped_short = 0

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
                gen_text = ref_text or "This is a sample text for activation extraction"
            gen_text = sanitize_gen_text_for_single_batch(gen_text=gen_text, ref_text=ref_text)

            for layer_idx in selected_layers:
                block = runtime.model.transformer.transformer_blocks[layer_idx]
                block.step_residual_tokens = [None] * int(cfg.nfe_step)
                block._capture_call_counter = 0

            ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(str(wav_path), ref_text)
            ref_audio_len = estimate_ref_audio_token_len(str(ref_audio_processed))
            if cfg.debug_verbose:
                logger.info(
                    f"[提取-细节] 均值提取样本 | idx={i} | file={wav_path.name} | "
                    f"ref_tokens={ref_audio_len} | gen_text='{gen_text[:80]}'"
                )
            if ref_audio_len < int(cfg.min_ref_tokens):
                skipped_short += 1
                if skipped_short <= 20:
                    logger.warning(
                        f"[提取] 参考token过短，跳过样本 {wav_path.name} | "
                        f"ref_tokens={ref_audio_len} < min_ref_tokens={int(cfg.min_ref_tokens)}"
                    )
                continue
            set_runtime_context_for_all_blocks(runtime.model, nfe_step=cfg.nfe_step, ref_audio_len=ref_audio_len, cfg_batch_size=1)
            if cfg.debug_verbose:
                infer_seed = (cfg.sampling_seed + i - 1) if cfg.sampling_seed is not None else None
                log_infer_call_detail(
                    logger=logger,
                    stage="extract.mean_activation",
                    emotion=emotion_subdir,
                    file_id=file_id,
                    ref_audio=str(ref_audio_processed),
                    ref_tokens=int(ref_audio_len),
                    ref_text=ref_text_processed,
                    gen_text=gen_text,
                    nfe_step=cfg.nfe_step,
                    cfg_strength=cfg.cfg_strength,
                    sway_sampling_coef=cfg.sway_sampling_coef,
                    seed=infer_seed,
                    extra=f"idx={i}/{len(audio_files)}",
                )

            call_infer_process_with_retry(
                infer_process_fn=infer_process,
                call_kwargs={
                    "ref_audio": ref_audio_processed,
                    "ref_text": ref_text_processed,
                    "gen_text": gen_text,
                    "model_obj": runtime.model,
                    "vocoder": runtime.vocoder,
                    "mel_spec_type": runtime.vocoder_name,
                    "nfe_step": cfg.nfe_step,
                    "cfg_strength": cfg.cfg_strength,
                    "sway_sampling_coef": cfg.sway_sampling_coef,
                    "device": runtime.device,
                },
                seed=(cfg.sampling_seed + i - 1) if cfg.sampling_seed is not None else None,
                logger=logger,
            )

            for local_idx, layer_idx in enumerate(selected_layers):
                block = runtime.model.transformer.transformer_blocks[layer_idx]
                tokens = fill_missing_steps(block.step_residual_tokens or [], cfg.nfe_step)
                if tokens is None:
                    if cfg.debug_verbose:
                        logger.warning(
                            f"[提取-细节] 层无token捕获 | idx={i} | file={wav_path.name} | layer={layer_idx}"
                        )
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

            if cfg.debug_verbose:
                per_layer_states = ", ".join(
                    [f"L{li}:{layer_counts[idx]}" for idx, li in enumerate(selected_layers)]
                )
                logger.info(
                    f"[提取-细节] 累计计数 | idx={i} | file={wav_path.name} | {per_layer_states}"
                )

            if i % 10 == 0 or i == len(audio_files):
                logger.info(f"[提取] 进度 {emotion_subdir}: {i}/{len(audio_files)}")
    finally:
        disable_residual_capture(runtime.model)

    if not any(c > 0 for c in layer_counts):
        raise RuntimeError(
            f"[提取] {emotion_subdir} 完全未捕获到残差。"
            f" 可用样本={len(audio_files)} | skip_short={skipped_short}"
        )

    if skipped_short > 0:
        logger.info(f"[提取] 过滤统计 {emotion_subdir} | skip_short={skipped_short}")

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
    min_ref_text_en_words: int,
    min_ref_text_zh_chars: int,
    debug_verbose: bool,
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

    rng = random.Random(seed)

    refs_with_lang: List[Dict[str, str]] = []
    skipped_short_text = 0
    for wav in all_neutral:
        fid = wav.stem
        ref_text = transcription_map.get(fid, "")
        if is_ref_text_too_short(
            ref_text,
            min_en_words=int(min_ref_text_en_words),
            min_zh_chars=int(min_ref_text_zh_chars),
        ):
            skipped_short_text += 1
            continue
        refs_with_lang.append(
            {
                "file_id": fid,
                "ref_audio": str(wav),
                "ref_text": ref_text,
                "lang": "zh" if contains_cjk(ref_text) else "en",
            }
        )
    if skipped_short_text > 0:
        logger.info(
            f"[搜索样本] 过滤短文本参考: {skipped_short_text} "
            f"(min_ref_text_en_words={int(min_ref_text_en_words)}, "
            f"min_ref_text_zh_chars={int(min_ref_text_zh_chars)})"
        )
    if len(refs_with_lang) < num_refs:
        raise RuntimeError(
            f"[搜索样本] 可用参考不足（文本过滤后）: {len(refs_with_lang)} < {num_refs}. "
            "请放宽文本阈值或补充字幕质量。"
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
        _ = i  # 保留索引变量，便于后续扩展
        gen_text = sanitize_gen_text_for_single_batch(
            gen_text=select_search_gen_text_by_lang(item["lang"]),
            ref_text=item["ref_text"],
        )
        out.append(
            SearchReference(
                ref_audio=item["ref_audio"],
                ref_text=item["ref_text"],
                gen_text=gen_text,
                file_id=item["file_id"],
            )
        )
    logger.info(f"[搜索样本] 已构建 emotion2vec 搜索样本: {len(out)}")
    if debug_verbose:
        for idx, item in enumerate(selected[:num_refs], start=1):
            logger.info(
                f"[搜索样本-细节] ref#{idx:02d} | file_id={item['file_id']} | "
                f"lang={item['lang']} | ref_text='{str(item['ref_text'])[:80]}'"
            )
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
    min_search_ref_tokens: int,
    sampling_seed: Optional[int],
    debug_verbose: bool,
    logger: logging.Logger,
) -> torch.Tensor:
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    if not references:
        raise ValueError("emotion2vec 搜索样本不能为空。")

    canonical = normalize_emotion_label(target_emotion)
    target_idx = EMOTION_LABEL_TO_INDEX[canonical]
    target_idx_resolved = False

    score_vectors: List[torch.Tensor] = []
    for vec in layer_steering_vectors:
        vec = vec.to(runtime.device)
        score_vectors.append(vec)

    steps, num_tokens = score_vectors[0].shape[:2]
    for vec in score_vectors[1:]:
        if vec.shape[:2] != (steps, num_tokens):
            raise ValueError("各层 steering shape 不一致。")
    if debug_verbose:
        for li, vec in zip(selected_layers, score_vectors):
            v = vec.float()
            logger.info(
                f"[Token评分-细节] 层向量统计 | layer={li} | shape={tuple(v.shape)} | "
                f"min={float(v.min().item()):.6f} | max={float(v.max().item()):.6f} | "
                f"mean={float(v.mean().item()):.6f} | std={float(v.std().item()):.6f}"
            )

    processed_refs: List[Dict[str, Any]] = []
    skipped_short_refs = 0
    for ref in references:
        ref_audio_path, ref_text = preprocess_ref_audio_text(ref.ref_audio, ref.ref_text)
        ref_audio_len = estimate_ref_audio_token_len(str(ref_audio_path))
        if ref_audio_len < int(min_search_ref_tokens):
            skipped_short_refs += 1
            if skipped_short_refs <= 20:
                logger.warning(
                    f"[Token评分] 搜索参考过短，跳过: file_id={ref.file_id} | "
                    f"ref_tokens={ref_audio_len} < min_search_ref_tokens={int(min_search_ref_tokens)}"
                )
            continue
        safe_gen_text = sanitize_gen_text_for_single_batch(gen_text=ref.gen_text, ref_text=ref_text)
        processed_refs.append(
            {
                "ref_audio": ref_audio_path,
                "ref_text": ref_text,
                "gen_text": safe_gen_text,
                "file_id": ref.file_id,
                "ref_audio_len": ref_audio_len,
            }
        )

    if not processed_refs:
        raise RuntimeError(
            f"[Token评分] 没有可用搜索参考（全部被过滤或为空）。"
            f" refs_total={len(references)} | skipped_short_refs={skipped_short_refs}"
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="emosteer_ser_"))
    token_scores: List[float] = []
    logger.info(
        f"[Token评分] emotion2vec 全 token 搜索开始 | tokens={num_tokens} | refs={len(processed_refs)} | "
        f"skip_short_refs={skipped_short_refs} | min_search_ref_tokens={int(min_search_ref_tokens)}"
    )
    num_labels: Optional[int] = None

    try:
        for token_idx in range(num_tokens):
            single_token_scores: List[float] = []
            model_dtype = next(runtime.model.parameters()).dtype
            token_vecs = [layer[:, token_idx, :].to(dtype=model_dtype) for layer in score_vectors]
            if debug_verbose:
                norm_msg = ", ".join(
                    [
                        f"L{li}:{float(tv.float().norm(p=2).item()):.6f}"
                        for li, tv in zip(selected_layers, token_vecs)
                    ]
                )
                logger.info(f"[Token评分-细节] token={token_idx:04d} 各层向量范数 | {norm_msg}")

            for ref_idx, ref in enumerate(processed_refs):
                if debug_verbose:
                    logger.info(
                        f"[Token评分-细节] token={token_idx:04d} ref={ref_idx:02d} | "
                        f"file_id={ref['file_id']} | ref_tokens={ref['ref_audio_len']} | "
                        f"gen_text='{str(ref['gen_text'])[:80]}'"
                    )
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
                        if debug_verbose:
                            log_infer_call_detail(
                                logger=logger,
                                stage="build.token_scoring",
                                emotion=canonical,
                                file_id=str(ref["file_id"]),
                                ref_audio=str(ref["ref_audio"]),
                                ref_tokens=int(ref["ref_audio_len"]),
                                ref_text=str(ref["ref_text"]),
                                gen_text=str(ref["gen_text"]),
                                nfe_step=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                                seed=sampling_seed,
                                extra=f"token={token_idx:04d},ref_idx={ref_idx:02d}",
                            )
                        infer_output = call_infer_process_with_retry(
                            infer_process_fn=infer_process,
                            call_kwargs={
                                "ref_audio": ref["ref_audio"],
                                "ref_text": ref["ref_text"],
                                "gen_text": ref["gen_text"],
                                "model_obj": runtime.model,
                                "vocoder": runtime.vocoder,
                                "mel_spec_type": runtime.vocoder_name,
                                "show_info": (lambda *_args, **_kwargs: None),
                                "progress": None,
                                "nfe_step": nfe_step,
                                "cfg_strength": cfg_strength,
                                "sway_sampling_coef": sway_sampling_coef,
                                "device": runtime.device,
                            },
                            seed=sampling_seed,
                            logger=logger,
                        )
                        wav_np, sample_rate, _ = normalize_infer_output(infer_output)
                finally:
                    clear_token_steering(runtime.model, selected_layers)

                if wav_np is None:
                    single_token_scores.append(0.0)
                    if debug_verbose:
                        logger.warning(
                            f"[Token评分-细节] token={token_idx:04d} ref={ref_idx:02d} infer_output为空，记0分"
                        )
                    continue

                wav_path = temp_dir / f"token_{token_idx:04d}_ref_{ref_idx:02d}.wav"
                if isinstance(wav_np, torch.Tensor):
                    save_wav = wav_np.detach().to(torch.float32).cpu()
                else:
                    try:
                        save_wav = torch.from_numpy(wav_np).to(torch.float32)
                    except Exception:
                        save_wav = torch.as_tensor(wav_np, dtype=torch.float32)
                if save_wav.ndim == 1:
                    save_wav = save_wav.unsqueeze(0)
                torchaudio.save(str(wav_path), save_wav.cpu(), int(sample_rate))

                ser_result = ser_model.generate(str(wav_path), granularity="utterance", extract_embedding=False)
                if ser_result and len(ser_result) > 0:
                    labels = ser_result[0].get("labels", [])
                    if num_labels is None and isinstance(labels, list) and labels:
                        num_labels = len(labels)
                        logger.info(f"[Token评分-诊断] emotion2vec 标签数={num_labels} | labels={labels}")
                    if not target_idx_resolved:
                        target_idx = resolve_target_index_from_labels(labels, canonical)
                        mapped = labels[target_idx] if labels and len(labels) > target_idx else f"idx={target_idx}"
                        logger.info(f"[Token评分] 目标情绪 '{canonical}' 映射到标签 '{mapped}' (index={target_idx})")
                        target_idx_resolved = True
                    scores = ser_result[0].get("scores", [])
                    single_token_scores.append(float(scores[target_idx]) if len(scores) > target_idx else 0.0)
                    if debug_verbose:
                        top_idx = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else -1
                        top_label = labels[top_idx] if labels and 0 <= top_idx < len(labels) else "N/A"
                        top_score = float(scores[top_idx]) if scores and 0 <= top_idx < len(scores) else 0.0
                        tgt_score = float(scores[target_idx]) if len(scores) > target_idx else 0.0
                        score_sum = float(sum(scores)) if scores else 0.0
                        logger.info(
                            f"[Token评分-细节] token={token_idx:04d} ref={ref_idx:02d} | "
                            f"target_score={tgt_score:.6f} | top1={top_label}:{top_score:.6f} | "
                            f"sum_scores={score_sum:.6f}"
                        )
                else:
                    single_token_scores.append(0.0)
                    if debug_verbose:
                        logger.warning(
                            f"[Token评分-细节] token={token_idx:04d} ref={ref_idx:02d} emotion2vec无结果，记0分"
                        )

            token_scores.append(sum(single_token_scores) / max(1, len(single_token_scores)))
            if debug_verbose:
                logger.info(
                    f"[Token评分-逐token] token={token_idx:04d} | refs={len(single_token_scores)} | "
                    f"mean_score={token_scores[-1]:.8f} | per_ref_scores={[float(x) for x in single_token_scores]}"
                )
            if (token_idx + 1) % 10 == 0 or token_idx == num_tokens - 1:
                logger.info(
                    f"[Token评分] 进度 {token_idx + 1}/{num_tokens} | "
                    f"当前token均分={token_scores[-1]:.6f}"
                )
    finally:
        clear_token_steering(runtime.model, selected_layers)
        shutil.rmtree(temp_dir, ignore_errors=True)

    token_scores_t = torch.tensor(token_scores, dtype=torch.float32, device=runtime.device)
    if token_scores_t.numel() > 0:
        tmin = float(torch.min(token_scores_t).item())
        tmax = float(torch.max(token_scores_t).item())
        tmean = float(torch.mean(token_scores_t).item())
        tstd = float(torch.std(token_scores_t).item()) if token_scores_t.numel() > 1 else 0.0
        logger.info(
            f"[Token评分-诊断] 分数统计 | min={tmin:.6f} | max={tmax:.6f} | "
            f"mean={tmean:.6f} | std={tstd:.6f}"
        )
        if num_labels is not None and num_labels > 0:
            uniform = 1.0 / float(num_labels)
            logger.info(f"[Token评分-诊断] 均匀分布基线(1/{num_labels})={uniform:.6f}")
            if abs(tmean - uniform) < 0.01 and (tmax - tmin) < 0.02:
                logger.warning(
                    "[Token评分-诊断] 分数整体接近均匀分布，情感注入信号较弱。"
                    "建议检查 target label 映射、参考样本质量与 alpha。"
                )
    return token_scores_t


def load_residual_pack(path: Path) -> Dict[str, Any]:
    pack = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(pack, dict) or "mean_residuals" not in pack:
        raise ValueError(f"残差文件格式非法: {path}")
    return pack


def load_steering_bundle(path: Path) -> Dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict):
        raise ValueError(f"steering bundle 格式非法(非dict): {path}")
    if "layers" not in bundle or "step_steering" not in bundle:
        raise ValueError(f"steering bundle 缺少关键字段 layers/step_steering: {path}")

    layers = bundle["layers"]
    steps = bundle["step_steering"]
    if not isinstance(layers, (list, tuple)) or not isinstance(steps, (list, tuple)):
        raise ValueError(f"steering bundle 字段类型非法: layers={type(layers)}, step_steering={type(steps)}")
    if len(layers) == 0 or len(steps) == 0 or len(layers) != len(steps):
        raise ValueError(
            f"steering bundle 层信息非法: len(layers)={len(layers)}, len(step_steering)={len(steps)}"
        )

    for i, x in enumerate(steps):
        if not torch.is_tensor(x):
            raise ValueError(f"step_steering[{i}] 不是张量: {type(x)}")
        if x.ndim not in (1, 2):
            raise ValueError(f"step_steering[{i}] 维度非法: {tuple(x.shape)}")
        if x.numel() == 0:
            raise ValueError(f"step_steering[{i}] 为空张量")
    return bundle


def resample_step_steering_to_nfe(step_steer: torch.Tensor, target_steps: int) -> torch.Tensor:
    target_steps = max(1, int(target_steps))
    if step_steer.ndim == 1:
        return step_steer.unsqueeze(0).repeat(target_steps, 1)
    if step_steer.shape[0] == target_steps:
        return step_steer
    x = step_steer.float().transpose(0, 1).unsqueeze(0)  # [1, D, S]
    x = F.interpolate(x, size=target_steps, mode="nearest")
    x = x.squeeze(0).transpose(0, 1).contiguous()
    return x.to(dtype=step_steer.dtype)


def to_mono_tensor_for_save(wav: Any) -> torch.Tensor:
    if isinstance(wav, torch.Tensor):
        t = wav.detach().to(torch.float32).cpu()
    else:
        try:
            t = torch.from_numpy(wav).to(torch.float32)
        except Exception:
            t = torch.as_tensor(wav, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim >= 2:
        if t.shape[0] > 2 and t.shape[1] <= 2:
            t = t.transpose(0, 1)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if t.shape[0] > 1:
            t = t[:1, :]
    return t.contiguous()


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
    if cfg.debug_verbose:
        logger.info(
            f"[构建-细节] 输入残差文件 | neutral={neutral_residual_file} | emotion={emotion_residual_file}"
        )
        logger.info(
            f"[构建-细节] neutral元信息 | target_len={neutral_pack.get('target_len', None)} | "
            f"num_samples={neutral_pack.get('num_samples', None)} | "
            f"layer_counts={neutral_pack.get('layer_counts', None)}"
        )
        logger.info(
            f"[构建-细节] emotion元信息 | target_len={emotion_pack.get('target_len', None)} | "
            f"num_samples={emotion_pack.get('num_samples', None)} | "
            f"layer_counts={emotion_pack.get('layer_counts', None)}"
        )

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
    if cfg.debug_verbose:
        logger.info(f"[构建-细节] 可用层数量={len(active_layers)} | layers={active_layers}")

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
        min_ref_text_en_words=cfg.min_ref_text_en_words,
        min_ref_text_zh_chars=cfg.min_ref_text_zh_chars,
        debug_verbose=cfg.debug_verbose,
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
        min_search_ref_tokens=cfg.min_search_ref_tokens,
        sampling_seed=cfg.sampling_seed,
        debug_verbose=cfg.debug_verbose,
        logger=logger,
    )

    if not torch.isfinite(token_importance).all():
        raise RuntimeError("[构建] token 分数出现 NaN/Inf。")
    if cfg.debug_verbose:
        ti = token_importance.float()
        logger.info(
            f"[构建-细节] token_importance统计 | len={ti.numel()} | "
            f"min={float(ti.min().item()):.6f} | max={float(ti.max().item()):.6f} | "
            f"mean={float(ti.mean().item()):.6f} | std={float(ti.std().item() if ti.numel()>1 else 0.0):.6f}"
        )

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
    if cfg.debug_verbose:
        logger.info(
            f"[构建-细节] top_k选择 | k={int(top_k)} | top_indices前10={top_indices[:10].tolist()} | "
            f"top_values前10={[float(x) for x in top_values[:10].tolist()]}"
        )
        logger.info(
            f"[构建-细节] softmax权重前10={[float(x) for x in weights[:10].tolist()]} | "
            f"weight_sum={float(weights.sum().item()):.6f}"
        )

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
        if cfg.debug_verbose:
            logger.info(
                f"[构建-细节] 层step_steer统计 | layer={active_layers[len(steering_steps)]} | "
                f"shape={tuple(step_steer.shape)} | min={float(step_steer.min().item()):.6f} | "
                f"max={float(step_steer.max().item()):.6f} | mean={float(step_steer.mean().item()):.6f} | "
                f"std={float(step_steer.std().item() if step_steer.numel()>1 else 0.0):.6f}"
            )
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
            "min_search_ref_tokens": int(cfg.min_search_ref_tokens),
            "min_ref_text_en_words": int(cfg.min_ref_text_en_words),
            "min_ref_text_zh_chars": int(cfg.min_ref_text_zh_chars),
            "debug_verbose": bool(cfg.debug_verbose),
        },
    }
    torch.save(bundle, output_file)

    logger.info(f"[构建] 已保存 steering bundle: {output_file}")
    logger.info(f"[构建] 生效层: {active_layers}")
    logger.info(f"[构建] top-k 数量: {len(top_indices)}")
    return bundle


def run_convert_stage(
    runtime: RuntimeHandles,
    args: argparse.Namespace,
    default_steering_out: Path,
    logger: logging.Logger,
) -> None:
    cfg = ConvertConfig(
        steering_bundle=args.steering_bundle,
        ref_audio=args.ref_audio,
        ref_text=str(args.ref_text or ""),
        gen_text=str(args.gen_text or ""),
        alpha=float(args.alpha),
        output_wav=args.output_wav,
        nfe_step_override=int(args.convert_nfe_step),
        cfg_strength=float(args.cfg_strength),
        sway_sampling_coef=float(args.sway_sampling_coef),
        sampling_seed=args.seed,
    )

    steering_bundle_path = (cfg.steering_bundle or default_steering_out).resolve()
    if not steering_bundle_path.exists():
        raise FileNotFoundError(
            f"[Convert] 未找到 steering bundle: {steering_bundle_path}。"
            "请先运行 build 阶段，或用 --steering_bundle 显式指定。"
        )
    if cfg.ref_audio is None:
        raise ValueError("[Convert] 缺少 --ref_audio。")
    ref_audio = cfg.ref_audio.resolve()
    if not ref_audio.exists():
        raise FileNotFoundError(f"[Convert] ref_audio 不存在: {ref_audio}")
    if not cfg.gen_text.strip():
        raise ValueError("[Convert] 缺少 --gen_text（不能为空）。")

    output_wav = cfg.output_wav
    if output_wav is None:
        emo = safe_name(args.emotion or "emotion")
        output_wav = args.output_dir / f"convert_{emo}.wav"
    output_wav = output_wav.resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_steering_bundle(steering_bundle_path)
    layers = [int(x) for x in bundle["layers"]]
    step_steering_raw: Sequence[torch.Tensor] = bundle["step_steering"]

    default_nfe = int(bundle.get("expected_nfe_step", bundle.get("num_steps", args.nfe_step)))
    if default_nfe <= 0:
        default_nfe = int(args.nfe_step)
    nfe_step = int(cfg.nfe_step_override) if int(cfg.nfe_step_override) > 0 else int(default_nfe)

    step_steering: List[torch.Tensor] = []
    model_dtype = next(runtime.model.parameters()).dtype
    step_shape_logs: List[str] = []
    for li, sv in zip(layers, step_steering_raw):
        sv2 = resample_step_steering_to_nfe(sv, nfe_step)
        step_steering.append(sv2.to(dtype=model_dtype))
        step_shape_logs.append(f"L{li}:{tuple(sv.shape)}->{tuple(sv2.shape)}")

    logger.info(
        f"[Convert] 加载 steering bundle: {steering_bundle_path} | layers={layers} | "
        f"alpha={cfg.alpha:.4f} | nfe_step={nfe_step}"
    )
    logger.info(f"[Convert] step_steering 形状对齐: {', '.join(step_shape_logs)}")

    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(str(ref_audio), cfg.ref_text)
    ref_audio_len = estimate_ref_audio_token_len(str(ref_audio_processed))
    logger.info(
        f"[Convert] 参考音频预处理完成 | ref_audio={ref_audio_processed} | "
        f"ref_tokens={ref_audio_len} | ref_text_len={len(str(ref_text_processed))}"
    )
    if ref_audio_len <= 0:
        raise RuntimeError("[Convert] 参考音频 token 长度估计失败。")

    set_runtime_context_for_all_blocks(
        runtime.model,
        nfe_step=nfe_step,
        ref_audio_len=ref_audio_len,
        cfg_batch_size=1,
    )
    set_token_steering(
        runtime.model,
        selected_layers=layers,
        per_layer_token_step_vecs=step_steering,
        alpha=cfg.alpha,
        nfe_step=nfe_step,
        ref_audio_len=ref_audio_len,
    )

    try:
        if bool(args.debug_verbose):
            log_infer_call_detail(
                logger=logger,
                stage="convert",
                emotion=str(args.emotion),
                file_id=str(Path(str(ref_audio_processed)).stem),
                ref_audio=str(ref_audio_processed),
                ref_tokens=int(ref_audio_len),
                ref_text=str(ref_text_processed),
                gen_text=str(cfg.gen_text),
                nfe_step=nfe_step,
                cfg_strength=cfg.cfg_strength,
                sway_sampling_coef=cfg.sway_sampling_coef,
                seed=cfg.sampling_seed,
                extra=f"layers={layers},alpha={cfg.alpha:.4f}",
            )
        infer_output = call_infer_process_compat(
            infer_process_fn=infer_process,
            call_kwargs={
                "ref_audio": ref_audio_processed,
                "ref_text": ref_text_processed,
                "gen_text": cfg.gen_text,
                "model_obj": runtime.model,
                "vocoder": runtime.vocoder,
                "mel_spec_type": runtime.vocoder_name,
                "show_info": (lambda *_args, **_kwargs: None),
                "progress": None,
                "nfe_step": nfe_step,
                "cfg_strength": cfg.cfg_strength,
                "sway_sampling_coef": cfg.sway_sampling_coef,
                "device": runtime.device,
            },
            seed=cfg.sampling_seed,
            logger=logger,
        )
    finally:
        clear_token_steering(runtime.model, layers)

    wav_out, sample_rate, _ = normalize_infer_output(infer_output)
    if wav_out is None:
        raise RuntimeError("[Convert] infer_process 未返回音频。")
    wav_tensor = to_mono_tensor_for_save(wav_out)
    torchaudio.save(str(output_wav), wav_tensor, int(sample_rate))

    logger.info(
        f"[Convert] 合成完成 | sample_rate={int(sample_rate)} | num_frames={int(wav_tensor.shape[-1])}"
    )
    logger.info(f"[Convert] 已保存输出音频: {output_wav}")


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
        min_ref_tokens=args.min_ref_tokens,
        target_len_mode=args.target_len_mode,
        debug_verbose=bool(args.debug_verbose),
        min_ref_text_en_words=int(args.min_ref_text_en_words),
        min_ref_text_zh_chars=int(args.min_ref_text_zh_chars),
        prune_bad_samples=bool(args.prune_bad_samples),
    )

    speaker_filter = parse_speaker_filter(cfg.speaker_filter)
    trans_map = load_transcription_map(cfg.dataset_dir)
    if cfg.prune_bad_samples:
        logger.info(
            "[清理] 启用数据清理：将删除 ref_tokens 过短或 ref_text 过短的样本（直接改写数据集）"
        )
        prune_bad_samples_from_dataset(
            dataset_dir=cfg.dataset_dir,
            emotion_subdir=cfg.neutral,
            transcription_map=trans_map,
            min_ref_tokens=cfg.min_ref_tokens,
            min_ref_text_en_words=cfg.min_ref_text_en_words,
            min_ref_text_zh_chars=cfg.min_ref_text_zh_chars,
            logger=logger,
        )
        prune_bad_samples_from_dataset(
            dataset_dir=cfg.dataset_dir,
            emotion_subdir=cfg.emotion,
            transcription_map=trans_map,
            min_ref_tokens=cfg.min_ref_tokens,
            min_ref_text_en_words=cfg.min_ref_text_en_words,
            min_ref_text_zh_chars=cfg.min_ref_text_zh_chars,
            logger=logger,
        )
        trans_map = load_transcription_map(cfg.dataset_dir)

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
        raise RuntimeError(
            f"[提取] neutral 无可用音频。已检查: {cfg.dataset_dir / 'audio' / cfg.neutral} "
            f"和 {cfg.dataset_dir / cfg.neutral}"
        )
    if not emotion_files:
        raise RuntimeError(
            f"[提取] emotion 无可用音频。已检查: {cfg.dataset_dir / 'audio' / cfg.emotion} "
            f"和 {cfg.dataset_dir / cfg.emotion}"
        )

    logger.info(
        f"[提取] 样本统计 | neutral={len(neutral_files)} | emotion={len(emotion_files)} | "
        f"max_samples={cfg.max_samples}"
    )

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
        min_ref_tokens=cfg.min_ref_tokens,
        min_ref_text_en_words=cfg.min_ref_text_en_words,
        min_ref_text_zh_chars=cfg.min_ref_text_zh_chars,
        fast_ref_only=(cfg.target_len_mode == "ref_audio"),
        debug_verbose=cfg.debug_verbose,
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
        min_ref_tokens=cfg.min_ref_tokens,
        min_ref_text_en_words=cfg.min_ref_text_en_words,
        min_ref_text_zh_chars=cfg.min_ref_text_zh_chars,
        fast_ref_only=(cfg.target_len_mode == "ref_audio"),
        debug_verbose=cfg.debug_verbose,
        transcription_map=trans_map,
        logger=logger,
    )
    target_len = max(1, int(round((target_len_n + target_len_e) / 2)))
    target_len_source = "ref_audio_token_len_avg" if cfg.target_len_mode == "ref_audio" else "captured_first_residual_avg"
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

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help=(
            "数据集根目录。支持两种结构："
            "1) 旧版 <root>/<emotion>/*.wav + transcription/*.txt；"
            "2) merged版 <root>/audio/<emotion>/*.wav + subtitles/subtitles.jsonl"
        ),
    )
    parser.add_argument("--emotion", type=str, required=True, help="目标情绪目录名，例如 sad")
    parser.add_argument("--neutral", type=str, default="neutral", help="中性目录名")
    parser.add_argument("--speaker_filter", type=str, default=None, help="说话人筛选，如 0010-0015 或 0010,0011")
    parser.add_argument("--max_samples", type=int, default=250, help="每个情绪用于残差提取的最大样本数")
    parser.add_argument("--search_samples", type=int, default=10, help="emotion2vec token 搜索的参考样本数")
    parser.add_argument(
        "--min_search_ref_tokens",
        type=int,
        default=32,
        help="token评分阶段最小参考token长度，低于该值的搜索参考会被跳过",
    )

    parser.add_argument("--layers", type=str, default="paper", help="层选择: paper/all/逗号分隔索引")
    parser.add_argument("--top_k", type=int, default=200, help="top-k token 数量")
    parser.add_argument("--seed", type=int, default=None, help="全流程随机种子")
    parser.add_argument("--text_mode", type=str, choices=["random_pool", "ref_text"], default="random_pool")
    parser.add_argument("--text_seed", type=int, default=1234)
    parser.add_argument(
        "--target_len_mode",
        type=str,
        choices=["ref_audio", "captured"],
        default="ref_audio",
        help="目标token长度估计方式：ref_audio(仅按预处理参考音频估计，不跑inference) 或 captured(按捕获残差估计)",
    )

    parser.add_argument("--nfe_step", type=int, default=32)
    parser.add_argument("--min_ref_tokens", type=int, default=32, help="提取阶段最小参考token长度，低于该值样本会被跳过")
    parser.add_argument("--min_ref_text_en_words", type=int, default=3, help="最小英文参考文本词数（低于阈值视为短文本）")
    parser.add_argument("--min_ref_text_zh_chars", type=int, default=6, help="最小中文参考文本字数（低于阈值视为短文本）")
    parser.add_argument(
        "--prune_bad_samples",
        action="store_true",
        help="在extract前直接删除数据集中 ref_tokens/ref_text 过短样本，并同步更新 subtitles/manifests",
    )
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--sway_sampling_coef", type=float, default=-1.0)
    parser.add_argument("--step_aggregation_mode", type=str, choices=["per_step", "mean_repeat"], default="per_step")
    parser.add_argument("--post_agg_norm", action="store_true")
    parser.add_argument(
        "--debug_verbose",
        action="store_true",
        help="开启详细诊断日志（用于排查提取/构建/打分阶段问题）",
    )
    parser.add_argument("--steering_bundle", type=Path, default=None, help="convert阶段使用的 steering bundle(.pt)；默认使用输出目录中的 --steering_pt")
    parser.add_argument("--ref_audio", type=Path, default=None, help="convert阶段参考音频路径")
    parser.add_argument("--ref_text", type=str, default="", help="convert阶段参考音频对应文本")
    parser.add_argument("--gen_text", type=str, default="", help="convert阶段要合成的目标文本")
    parser.add_argument("--alpha", type=float, default=2.0, help="convert阶段 steering 注入强度")
    parser.add_argument("--convert_nfe_step", type=int, default=0, help="convert阶段ODE步数；<=0 时自动使用 bundle 里的 expected_nfe_step")
    parser.add_argument("--output_wav", type=Path, default=None, help="convert阶段输出音频路径")

    parser.add_argument("--model_name", type=str, default="F5TTS_v1_Base")
    parser.add_argument("--vocoder_name", type=str, default="vocos")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model_cfg", type=str, default=None)
    parser.add_argument("--ckpt_file", type=str, default=None)
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--load_vocoder_from_local", action="store_true")
    parser.add_argument("--f5_repo_dir", type=Path, default=None, help="可选：f5_tts 仓库根目录（脚本会自动加 /src 到 sys.path）")
    parser.add_argument(
        "--disable_auto_bootstrap_f5",
        action="store_true",
        help="关闭自动自举 f5_tts（默认开启）。关闭后若环境无 f5_tts 会直接报错。",
    )
    parser.add_argument(
        "--f5_vendor_dir",
        type=Path,
        default=None,
        help="自动自举时的目标目录。默认: <脚本目录>/_vendor/F5-TTS",
    )
    parser.add_argument(
        "--f5_git_url",
        type=str,
        default="https://github.com/SWivid/F5-TTS.git",
        help="自动自举使用的 F5-TTS 仓库地址",
    )
    parser.add_argument(
        "--f5_git_ref",
        type=str,
        default="main",
        help="自动自举使用的分支/标签/提交",
    )

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
        f"top_k={args.top_k} | max_samples={args.max_samples} | search_samples={args.search_samples} | "
        f"min_ref_tokens={args.min_ref_tokens} | min_search_ref_tokens={args.min_search_ref_tokens} | "
        f"min_ref_text_en_words={int(args.min_ref_text_en_words)} | "
        f"min_ref_text_zh_chars={int(args.min_ref_text_zh_chars)} | "
        f"prune_bad_samples={bool(args.prune_bad_samples)} | "
        f"target_len_mode={args.target_len_mode} | debug_verbose={bool(args.debug_verbose)}"
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
        auto_bootstrap_f5=not bool(args.disable_auto_bootstrap_f5),
        f5_vendor_dir=args.f5_vendor_dir,
        f5_git_url=str(args.f5_git_url),
        f5_git_ref=str(args.f5_git_ref),
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
            min_search_ref_tokens=args.min_search_ref_tokens,
            step_aggregation_mode=args.step_aggregation_mode,
            post_agg_norm=bool(args.post_agg_norm),
            debug_verbose=bool(args.debug_verbose),
            min_ref_text_en_words=int(args.min_ref_text_en_words),
            min_ref_text_zh_chars=int(args.min_ref_text_zh_chars),
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
        run_convert_stage(
            runtime=runtime,
            args=args,
            default_steering_out=steering_out,
            logger=logger,
        )

    logger.info("========== 全流程结束 ==========")
    logger.info(f"[日志] 已写入: {args.log_file}")


if __name__ == "__main__":
    main()
