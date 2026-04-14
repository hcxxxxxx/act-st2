#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen


DATASET_CHOICES = ["ravdess", "cremad", "tess", "asvp_esd", "emo_emilia"]

CANONICAL_EMOTION_ALIASES: Dict[str, str] = {
    "anger": "angry",
    "angry": "angry",
    "ang": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "dis": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "fea": "fear",
    "happy": "happy",
    "happiness": "happy",
    "hap": "happy",
    "neutral": "neutral",
    "neu": "neutral",
    "calm": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "pleasant_surprise": "surprise",
    "ps": "surprise",
    "sur": "surprise",
    "excited": "excited",
    "excitement": "excited",
    "pleasure": "pleasure",
    "pain": "pain",
    "disappointment": "disappointment",
    "boredom": "boredom",
}

COMMON_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}

RAVDESS_EMOTION_CODE = {
    "01": "neutral",
    "02": "neutral",  # calm -> neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}
RAVDESS_STATEMENT = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}

CREMAD_SENTENCE = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}
CREMAD_EMOTION_CODE = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

ASVP_EMOTION_CODE = {
    "01": "boredom",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
    "09": "excited",
    "10": "pleasure",
    "11": "pain",
    "12": "disappointment",
    "13": "breath",
}

TEXT_KEYS = {"text", "transcript", "sentence", "caption", "utterance", "asr", "content"}
EMOTION_KEYS = {"emotion", "emo", "label", "emotion_label", "category"}
PATH_KEYS = {
    "path",
    "audio",
    "audio_path",
    "wav",
    "wav_path",
    "file",
    "filename",
    "file_name",
    "utt_id",
    "uid",
    "id",
}


def setup_logger(log_file: Optional[Path]) -> logging.Logger:
    logger = logging.getLogger("intersection_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def run_cmd(cmd: Sequence[str], logger: logging.Logger, cwd: Optional[Path] = None) -> None:
    logger.info(f"[命令] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"命令执行失败({proc.returncode}): {' '.join(cmd)}")


def request_json(url: str) -> Dict[str, Any]:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dst: Path, logger: logging.Logger) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        logger.info(f"[下载] 跳过(已存在): {dst}")
        return

    if shutil.which("aria2c"):
        cmd = [
            "aria2c",
            "-x",
            "16",
            "-s",
            "16",
            "-c",
            "-d",
            str(dst.parent),
            "-o",
            dst.name,
            url,
        ]
        run_cmd(cmd, logger)
        return

    if shutil.which("curl"):
        cmd = ["curl", "-L", "--retry", "5", "-C", "-", "-o", str(dst), url]
        run_cmd(cmd, logger)
        return

    logger.info(f"[下载] 使用 urllib: {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, dst.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def extract_archive(archive_path: Path, target_dir: Path, logger: logging.Logger) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    logger.info(f"[解压] {archive_path.name} -> {target_dir}")
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
        return

    if suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(target_dir)
        return

    if suffix == ".rar":
        if shutil.which("unrar"):
            run_cmd(["unrar", "x", "-o+", str(archive_path), str(target_dir)], logger)
            return
        if shutil.which("7z"):
            run_cmd(["7z", "x", "-y", f"-o{str(target_dir)}", str(archive_path)], logger)
            return
        raise RuntimeError("检测到 .rar 文件，但未找到 unrar/7z。请先安装后重试。")

    logger.warning(f"[解压] 未识别压缩格式，跳过: {archive_path}")


def list_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in COMMON_AUDIO_SUFFIXES:
            files.append(p)
    return files


def normalize_emotion(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    s = label.strip().lower().replace("-", "_").replace(" ", "_")
    if s in CANONICAL_EMOTION_ALIASES:
        return CANONICAL_EMOTION_ALIASES[s]
    s2 = s.replace("_", "")
    for k, v in CANONICAL_EMOTION_ALIASES.items():
        if k.replace("_", "") == s2:
            return v
    return None


def infer_emotion_from_path(path: Path) -> Optional[str]:
    tokens = re.split(r"[^a-zA-Z0-9]+", path.as_posix().lower())
    for token in tokens:
        emo = normalize_emotion(token)
        if emo:
            return emo
    return None


def stable_uid(dataset_name: str, src_path: Path) -> str:
    h = hashlib.sha1(f"{dataset_name}:{src_path.as_posix()}".encode("utf-8")).hexdigest()[:12]
    return f"{dataset_name}_{h}"


def load_metadata_lookup(root: Path, logger: logging.Logger) -> Dict[str, Dict[str, str]]:
    """
    返回: key(文件名/stem/相对路径) -> {"text": "...", "emotion": "..."}
    """
    lookup: Dict[str, Dict[str, str]] = {}

    def put_key(k: str, text: Optional[str], emotion: Optional[str]) -> None:
        if not k:
            return
        key = k.replace("\\", "/").strip()
        if not key:
            return
        obj = lookup.setdefault(key, {})
        if text and "text" not in obj:
            obj["text"] = text.strip()
        if emotion and "emotion" not in obj:
            obj["emotion"] = emotion.strip()
        stem = Path(key).stem
        base = Path(key).name
        for kk in [stem, base]:
            o2 = lookup.setdefault(kk, {})
            if text and "text" not in o2:
                o2["text"] = text.strip()
            if emotion and "emotion" not in o2:
                o2["emotion"] = emotion.strip()

    def find_first(row: Dict[str, Any], keys: Set[str]) -> Optional[str]:
        for k, v in row.items():
            if k.lower() in keys and v is not None:
                return str(v)
        return None

    small_text_meta = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".csv", ".tsv", ".jsonl", ".json", ".txt", ".parquet"}:
            # 只读取元数据候选，避免误扫超大语音文件
            if p.stat().st_size <= 512 * 1024 * 1024:
                small_text_meta.append(p)

    for meta_file in small_text_meta:
        suffix = meta_file.suffix.lower()
        try:
            if suffix in {".csv", ".tsv"}:
                delim = "," if suffix == ".csv" else "\t"
                with meta_file.open("r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f, delimiter=delim)
                    if not reader.fieldnames:
                        continue
                    for row in reader:
                        path_val = find_first(row, PATH_KEYS)
                        text_val = find_first(row, TEXT_KEYS)
                        emo_val = find_first(row, EMOTION_KEYS)
                        if path_val:
                            put_key(path_val, text_val, emo_val)
            elif suffix == ".jsonl":
                with meta_file.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            continue
                        path_val = find_first(row, PATH_KEYS)
                        text_val = find_first(row, TEXT_KEYS)
                        emo_val = find_first(row, EMOTION_KEYS)
                        if path_val:
                            put_key(path_val, text_val, emo_val)
            elif suffix == ".json":
                obj = json.loads(meta_file.read_text(encoding="utf-8", errors="ignore"))
                rows = obj if isinstance(obj, list) else [obj]
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    path_val = find_first(row, PATH_KEYS)
                    text_val = find_first(row, TEXT_KEYS)
                    emo_val = find_first(row, EMOTION_KEYS)
                    if path_val:
                        put_key(path_val, text_val, emo_val)
            elif suffix == ".txt":
                with meta_file.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) >= 2:
                            put_key(parts[0], parts[1], None)
            elif suffix == ".parquet":
                try:
                    import pyarrow.parquet as pq  # type: ignore
                except Exception:
                    continue
                pf = pq.ParquetFile(str(meta_file))
                for i in range(pf.num_row_groups):
                    table = pf.read_row_group(i)
                    cols = [c.lower() for c in table.column_names]
                    path_col = next((c for c in table.column_names if c.lower() in PATH_KEYS), None)
                    text_col = next((c for c in table.column_names if c.lower() in TEXT_KEYS), None)
                    emo_col = next((c for c in table.column_names if c.lower() in EMOTION_KEYS), None)
                    if path_col is None:
                        continue
                    paths = table[path_col].to_pylist()
                    texts = table[text_col].to_pylist() if text_col else [None] * len(paths)
                    emos = table[emo_col].to_pylist() if emo_col else [None] * len(paths)
                    for p, t, e in zip(paths, texts, emos):
                        if p is None:
                            continue
                        put_key(str(p), str(t) if t is not None else None, str(e) if e is not None else None)
        except Exception as e:
            logger.warning(f"[元数据] 解析失败，已跳过 {meta_file}: {e}")

    logger.info(f"[元数据] {root.name}: 构建映射条目 {len(lookup)}")
    return lookup


def find_meta_for_audio(meta_lookup: Dict[str, Dict[str, str]], audio_path: Path, root: Path) -> Dict[str, str]:
    rel = audio_path.relative_to(root).as_posix()
    candidates = [rel, audio_path.name, audio_path.stem]
    for c in candidates:
        if c in meta_lookup:
            return meta_lookup[c]
    return {}


def ensure_ravdess_audio_ready(data_dir: Path, logger: logging.Logger) -> None:
    if any(data_dir.rglob("*.wav")):
        return
    zip_candidates = [
        data_dir / "Audio_Speech_Actors_01-24.zip",
        *(data_dir.rglob("Audio_Speech_Actors_01-24.zip")),
    ]
    zip_path = next((z for z in zip_candidates if z.exists()), None)
    if zip_path is None:
        logger.warning("[RAVDESS] 未找到 wav，也未找到 Audio_Speech_Actors_01-24.zip")
        return
    logger.info(f"[RAVDESS] 检测到未解压 zip，开始自动解压: {zip_path}")
    extract_archive(zip_path, data_dir, logger)


def resolve_emo_emilia_audio_path(data_dir: Path, wav_field: str) -> Optional[Path]:
    wav_field = str(wav_field).strip()
    if not wav_field:
        return None
    p = Path(wav_field)
    cands: List[Path] = []
    if p.is_absolute():
        cands.append(p)
    cands.append(data_dir / wav_field)
    cands.append(data_dir / wav_field.lstrip("./"))
    cands.append(data_dir / wav_field.replace("./Emo-Emilia/", ""))
    cands.append(data_dir / wav_field.replace("Emo-Emilia/", ""))
    cands.append(data_dir / "wav" / Path(wav_field).name)
    for c in cands:
        if c.exists():
            return c.resolve()
    return None


def make_cache_key(row: Dict[str, Any]) -> str:
    return f"{row.get('dataset','')}::{row.get('src_path','')}"


def load_cache_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for item in read_jsonl(path):
        k = item.get("key")
        if isinstance(k, str) and k:
            out[k] = item
    return out


def append_cache_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_quality_metrics(
    audio_path: Path,
    top_db: float = 40.0,
) -> Dict[str, float]:
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("需要 numpy 才能执行附录B音频质量筛选。") from e

    decode_errors: List[str] = []
    y = None
    sr = None

    # 1) wave 优先处理 wav，避免依赖外部解码后端
    if (y is None or sr is None) and audio_path.suffix.lower() in {".wav", ".wave"}:
        try:
            import wave

            with wave.open(str(audio_path), "rb") as wf:
                sr_w = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if sampwidth == 1:
                arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                arr = (arr - 128.0) / 128.0
            elif sampwidth == 2:
                arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            elif sampwidth == 3:
                b = np.frombuffer(raw, dtype=np.uint8)
                b = b.reshape(-1, 3)
                ints = (
                    b[:, 0].astype(np.int32)
                    | (b[:, 1].astype(np.int32) << 8)
                    | (b[:, 2].astype(np.int32) << 16)
                )
                ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
                arr = ints.astype(np.float32) / 8388608.0
            elif sampwidth == 4:
                arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                raise RuntimeError(f"不支持的 wav 采样宽度: {sampwidth} bytes")

            if n_channels > 1:
                arr = arr.reshape(-1, n_channels).mean(axis=1)

            y = np.asarray(arr, dtype=np.float32)
            sr = int(sr_w)
        except Exception as e:
            decode_errors.append(f"wave: {e}")

    # 2) soundfile 兜底（支持更多音频格式）
    if y is None or sr is None:
        try:
            import soundfile as sf  # type: ignore

            y_sf, sr_sf = sf.read(str(audio_path), always_2d=False)
            if hasattr(y_sf, "ndim") and y_sf.ndim > 1:
                y_sf = y_sf.mean(axis=1)
            y = np.asarray(y_sf, dtype=np.float32)
            sr = int(sr_sf)
        except Exception as e:
            decode_errors.append(f"soundfile.read: {e}")

    # 3) librosa 最后兜底（若环境可用）
    if y is None or sr is None:
        try:
            import librosa  # type: ignore

            y_lb, sr_lb = librosa.load(str(audio_path), sr=None, mono=True)
            y = np.asarray(y_lb, dtype=np.float32)
            sr = int(sr_lb)
        except Exception as e:
            decode_errors.append(f"librosa.load: {e}")

    if y is None or sr is None:
        raise RuntimeError(
            "音频解码失败，所有后端均不可用: "
            + " | ".join(decode_errors)
        )

    if y is None or len(y) == 0 or sr is None or sr <= 0:
        return {"duration_sec": 0.0, "silence_ratio": 1.0, "snr_db": -999.0}

    y = np.asarray(y, dtype=np.float32)
    duration_sec = float(len(y) / sr)

    frame_length = 2048
    hop_length = 512
    if len(y) <= frame_length:
        rms = np.asarray([float(np.sqrt(np.mean(np.square(y)) + 1e-12))], dtype=np.float32)
    else:
        n_frames = 1 + (len(y) - frame_length) // hop_length
        rms_vals = []
        for i in range(n_frames):
            st = i * hop_length
            ed = st + frame_length
            frame = y[st:ed]
            rms_vals.append(float(np.sqrt(np.mean(np.square(frame)) + 1e-12)))
        rms = np.asarray(rms_vals, dtype=np.float32)

    if rms.size == 0:
        silence_ratio = 1.0
        voiced = np.zeros(1, dtype=bool)
    else:
        max_rms = float(max(float(rms.max()), 1e-12))
        threshold = max_rms * (10.0 ** (-float(top_db) / 20.0))
        voiced = rms > threshold
        silence_ratio = float(1.0 - float(voiced.mean()))

    if voiced.size == 0 or not bool(voiced.any()):
        speech_power = 1e-12
        noise_power = float((y.astype("float64") ** 2).mean() + 1e-12)
    else:
        speech_mask = np.zeros(len(y), dtype=bool)
        for i, is_voiced in enumerate(voiced.tolist()):
            if not is_voiced:
                continue
            st = i * hop_length
            ed = min(st + frame_length, len(y))
            speech_mask[st:ed] = True
        speech = y[speech_mask]
        noise = y[~speech_mask]
        speech_power = float((speech.astype("float64") ** 2).mean() + 1e-12) if speech.size else 1e-12
        if noise.size:
            noise_power = float((noise.astype("float64") ** 2).mean() + 1e-12)
        else:
            noise_power = 1e-12

    snr_db = float(10.0 * math.log10(max(speech_power, 1e-12) / max(noise_power, 1e-12)))
    return {
        "duration_sec": duration_sec,
        "silence_ratio": silence_ratio,
        "snr_db": snr_db,
    }


def run_quality_filter(
    rows: List[Dict[str, Any]],
    min_duration_sec: float,
    max_duration_sec: float,
    max_silence_ratio: float,
    min_snr_db: float,
    top_db: float,
    quality_cache_file: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    cache = load_cache_map(quality_cache_file)
    kept: List[Dict[str, Any]] = []
    stats = {
        "total": len(rows),
        "drop_duration": 0,
        "drop_silence": 0,
        "drop_snr": 0,
        "drop_decode": 0,
        "drop_missing": 0,
        "kept": 0,
    }
    decode_warn_count = 0

    for r in rows:
        src = Path(r["src_path"])
        if not src.exists():
            stats["drop_missing"] += 1
            continue
        key = make_cache_key(r)
        rec = cache.get(key)
        if rec is None:
            try:
                m = compute_quality_metrics(src, top_db=top_db)
            except Exception as e:
                stats["drop_decode"] += 1
                if decode_warn_count < 20:
                    logger.warning(f"[筛选-质量] 解码失败，跳过: {src.name} | {e}")
                decode_warn_count += 1
                continue
            rec = {
                "key": key,
                "dataset": r.get("dataset", ""),
                "src_path": str(src),
                "top_db": float(top_db),
                **m,
            }
            append_cache_row(quality_cache_file, rec)
            cache[key] = rec

        duration = float(rec.get("duration_sec", 0.0))
        silence_ratio = float(rec.get("silence_ratio", 1.0))
        snr_db = float(rec.get("snr_db", -999.0))
        ok_duration = (duration >= min_duration_sec) and (duration <= max_duration_sec)
        ok_silence = (silence_ratio <= max_silence_ratio)
        ok_snr = (snr_db >= min_snr_db)

        if not ok_duration:
            stats["drop_duration"] += 1
            continue
        if not ok_silence:
            stats["drop_silence"] += 1
            continue
        if not ok_snr:
            stats["drop_snr"] += 1
            continue

        r2 = dict(r)
        r2["duration_sec"] = duration
        r2["silence_ratio"] = silence_ratio
        r2["snr_db"] = snr_db
        kept.append(r2)

    stats["kept"] = len(kept)
    logger.info(
        "[筛选-质量] total=%d kept=%d drop_duration=%d drop_silence=%d drop_snr=%d drop_decode=%d drop_missing=%d",
        stats["total"],
        stats["kept"],
        stats["drop_duration"],
        stats["drop_silence"],
        stats["drop_snr"],
        stats["drop_decode"],
        stats["drop_missing"],
    )
    if stats["drop_decode"] > 0:
        logger.warning(f"[筛选-质量] 解码失败总数: {stats['drop_decode']}（仅前20条打印详细警告）")
    return kept, stats


def normalize_ser_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    s = str(label).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    # emotion2vec 常见标签兼容
    s = s.replace("fearful", "fear").replace("disgusted", "disgust").replace("surprised", "surprise")
    return normalize_emotion(s)


def run_ser_filter(
    rows: List[Dict[str, Any]],
    ser_model_name: str,
    ser_hub: str,
    ser_min_conf: float,
    ser_cache_file: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    try:
        from funasr import AutoModel  # type: ignore
    except Exception as e:
        raise RuntimeError("需要安装 funasr 才能执行 emotion2Vec 置信度筛选。") from e

    cache = load_cache_map(ser_cache_file)
    logger.info(f"[筛选-SER] 加载 emotion2vec: model={ser_model_name}, hub={ser_hub}")
    ser_model = AutoModel(model=ser_model_name, hub=ser_hub)

    kept: List[Dict[str, Any]] = []
    stats = {
        "total": len(rows),
        "drop_missing": 0,
        "drop_conf": 0,
        "drop_label": 0,
        "drop_invalid": 0,
        "kept": 0,
    }

    for r in rows:
        src = Path(r["src_path"])
        if not src.exists():
            stats["drop_missing"] += 1
            continue

        gt = normalize_emotion(r.get("emotion"))
        if gt is None:
            stats["drop_invalid"] += 1
            continue

        key = make_cache_key(r)
        rec = cache.get(key)
        if rec is None:
            pred_label = None
            pred_conf = 0.0
            labels: List[str] = []
            try:
                result = ser_model.generate(str(src), granularity="utterance", extract_embedding=False)
                item: Dict[str, Any] = {}
                if isinstance(result, list) and result:
                    item = result[0] if isinstance(result[0], dict) else {}
                elif isinstance(result, dict):
                    item = result
                if item:
                    labels = [str(x) for x in item.get("labels", [])]
                    scores = [float(x) for x in item.get("scores", [])]
                    if scores:
                        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
                        pred_conf = float(scores[best_idx])
                        if labels and best_idx < len(labels):
                            pred_label = labels[best_idx]
                        else:
                            pred_label = str(best_idx)
            except Exception as e:
                logger.warning(f"[筛选-SER] 推理失败，样本跳过: {src.name} | {e}")

            rec = {
                "key": key,
                "dataset": r.get("dataset", ""),
                "src_path": str(src),
                "pred_label_raw": pred_label or "",
                "pred_conf": float(pred_conf),
                "labels": labels,
            }
            append_cache_row(ser_cache_file, rec)
            cache[key] = rec

        pred = normalize_ser_label(rec.get("pred_label_raw"))
        conf = float(rec.get("pred_conf", 0.0))

        if conf < ser_min_conf:
            stats["drop_conf"] += 1
            continue
        if pred != gt:
            stats["drop_label"] += 1
            continue

        r2 = dict(r)
        r2["ser_pred"] = pred
        r2["ser_conf"] = conf
        kept.append(r2)

    stats["kept"] = len(kept)
    logger.info(
        "[筛选-SER] total=%d kept=%d drop_conf=%d drop_label=%d drop_missing=%d drop_invalid=%d",
        stats["total"],
        stats["kept"],
        stats["drop_conf"],
        stats["drop_label"],
        stats["drop_missing"],
        stats["drop_invalid"],
    )
    return kept, stats


def download_ravdess(root: Path, logger: logging.Logger) -> Path:
    ds_root = root / "ravdess"
    raw_dir = ds_root / "raw"
    data_dir = ds_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if any(data_dir.rglob("*.wav")):
        logger.info("[下载] RAVDESS 已存在，跳过下载")
        return data_dir

    record = request_json("https://zenodo.org/api/records/1188976")
    files = record.get("files", [])
    speech = [f for f in files if str(f.get("key", "")).startswith("Audio_Speech_Actors_01-24")]
    if not speech:
        raise RuntimeError("未在 Zenodo 1188976 中找到 Audio_Speech_Actors_01-24 文件。")
    f = speech[0]
    file_url = f["links"]["self"]
    archive = raw_dir / f["key"]
    download_file(file_url, archive, logger)
    extract_archive(archive, data_dir, logger)
    return data_dir


def download_cremad(root: Path, logger: logging.Logger) -> Path:
    ds_root = root / "cremad"
    data_dir = ds_root / "data"
    if any(data_dir.rglob("*.wav")):
        logger.info("[下载] CREMA-D 已存在，跳过下载")
        return data_dir
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    if not shutil.which("git"):
        raise RuntimeError("未找到 git，无法下载 CREMA-D。")

    clone_target = data_dir
    if clone_target.exists() and not any(clone_target.iterdir()):
        clone_target.rmdir()

    urls = [
        "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git",
        "https://github.com/CheyneyComputerScience/CREMA-D.git",
    ]
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            run_cmd(["git", "clone", "--depth", "1", url, str(clone_target)], logger)
            last_err = None
            break
        except Exception as e:
            logger.warning(f"[下载] CREMA-D 克隆失败 {url}: {e}")
            last_err = e
    if last_err is not None:
        raise last_err

    if shutil.which("git"):
        try:
            run_cmd(["git", "-C", str(clone_target), "lfs", "install"], logger)
            run_cmd(["git", "-C", str(clone_target), "lfs", "pull"], logger)
        except Exception:
            logger.warning("[下载] git-lfs 拉取失败，若音频是占位符，请安装 git-lfs 后手动执行 git lfs pull。")
    return data_dir


def download_tess(root: Path, logger: logging.Logger) -> Path:
    ds_root = root / "tess"
    raw_dir = ds_root / "raw"
    data_dir = ds_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if any(data_dir.rglob("*.wav")):
        logger.info("[下载] TESS 已存在，跳过下载")
        return data_dir

    archive = raw_dir / "tess_dataset.zip"
    url_direct = "https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF"
    try:
        download_file(url_direct, archive, logger)
        extract_archive(archive, data_dir, logger)
        return data_dir
    except Exception as e:
        logger.warning(f"[下载] TESS 直接打包下载失败，尝试逐文件下载: {e}")

    pid = quote("doi:10.5683/SP2/E8H2MF", safe="")
    meta_url = f"https://borealisdata.ca/api/datasets/:persistentId/?persistentId={pid}"
    meta = request_json(meta_url)
    files = (
        meta.get("data", {})
        .get("latestVersion", {})
        .get("files", [])
    )
    if not files:
        raise RuntimeError("TESS 元数据中未找到可下载文件。")

    for item in files:
        data_file = item.get("dataFile", {})
        file_id = data_file.get("id")
        file_name = data_file.get("filename", f"{file_id}.bin")
        if not file_id:
            continue
        file_url = f"https://borealisdata.ca/api/access/datafile/{file_id}"
        dst = raw_dir / file_name
        download_file(file_url, dst, logger)
        if dst.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".rar"}:
            extract_archive(dst, data_dir, logger)
    return data_dir


def download_asvp_esd(root: Path, logger: logging.Logger) -> Path:
    ds_root = root / "asvp_esd"
    raw_dir = ds_root / "raw"
    data_dir = ds_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if any(data_dir.rglob("*.wav")):
        logger.info("[下载] ASVP-ESD 已存在，跳过下载")
        return data_dir

    # 使用较新的公开记录（v2 扩展版）
    record = request_json("https://zenodo.org/api/records/7132783")
    files = record.get("files", [])
    if not files:
        raise RuntimeError("ASVP-ESD Zenodo 记录未返回文件。")
    for f in files:
        key = str(f.get("key", "asvp_file"))
        dst = raw_dir / key
        download_file(f["links"]["self"], dst, logger)
        if dst.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".rar"}:
            extract_archive(dst, data_dir, logger)
    return data_dir


def download_emo_emilia(root: Path, logger: logging.Logger, hf_token: Optional[str]) -> Path:
    ds_root = root / "emo_emilia"
    data_dir = ds_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if any(data_dir.rglob("*")):
        logger.info("[下载] Emo-Emilia 目录非空，默认跳过下载")
        return data_dir

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        raise RuntimeError("未安装 huggingface_hub，无法下载 Emo-Emilia。请先 pip install huggingface_hub") from e

    logger.info("[下载] 开始下载 Emo-Emilia (Hugging Face)")
    snapshot_download(
        repo_id="ASLP-lab/Emo-Emilia",
        repo_type="dataset",
        local_dir=str(data_dir),
        token=hf_token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return data_dir


def index_ravdess(data_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    ensure_ravdess_audio_ready(data_dir, logger)
    out: List[Dict[str, Any]] = []
    for wav in data_dir.rglob("*.wav"):
        stem = wav.stem
        parts = stem.split("-")
        if len(parts) != 7:
            continue
        if parts[0] != "03" or parts[1] != "01":
            # 仅使用 audio-only speech
            continue
        emo = RAVDESS_EMOTION_CODE.get(parts[2])
        if not emo:
            continue
        text = RAVDESS_STATEMENT.get(parts[4], "")
        out.append(
            {
                "dataset": "ravdess",
                "src_path": str(wav.resolve()),
                "emotion": emo,
                "emotion_raw": parts[2],
                "text": text,
                "speaker": parts[6],
                "uid": stable_uid("ravdess", wav),
            }
        )
    logger.info(f"[索引] RAVDESS 样本: {len(out)}")
    return out


def index_cremad(data_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    wav_roots = []
    for cand in [data_dir / "CREMA-D" / "AudioWAV", data_dir / "AudioWAV", data_dir]:
        if cand.exists():
            wav_roots.append(cand)
    wav_candidates: List[Path] = []
    seen: Set[str] = set()
    for root in wav_roots:
        for w in root.rglob("*.wav"):
            k = str(w.resolve())
            if k not in seen:
                seen.add(k)
                wav_candidates.append(w)
    for wav in wav_candidates:
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        actor = parts[0]
        sent_code = parts[1].upper()
        emo_code = parts[2].upper()
        emo = CREMAD_EMOTION_CODE.get(emo_code)
        if not emo:
            emo = normalize_emotion(emo_code)
        if not emo:
            continue
        text = CREMAD_SENTENCE.get(sent_code, "")
        out.append(
            {
                "dataset": "cremad",
                "src_path": str(wav.resolve()),
                "emotion": emo,
                "emotion_raw": emo_code,
                "text": text,
                "speaker": actor,
                "uid": stable_uid("cremad", wav),
            }
        )
    logger.info(f"[索引] CREMA-D 样本: {len(out)}")
    return out


def index_tess(data_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for wav in data_dir.rglob("*.wav"):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        # e.g. OAF_back_angry / YAF_word_pleasant_surprise
        if parts[-2].lower() == "pleasant" and parts[-1].lower() == "surprise":
            emo_raw = "pleasant_surprise"
            word = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
        else:
            emo_raw = parts[-1].lower()
            word = "_".join(parts[1:-1]) if len(parts) > 2 else ""
        emo = normalize_emotion(emo_raw)
        if not emo:
            continue
        text = word.replace("_", " ").strip()
        out.append(
            {
                "dataset": "tess",
                "src_path": str(wav.resolve()),
                "emotion": emo,
                "emotion_raw": emo_raw,
                "text": text if text else "",
                "speaker": parts[0],
                "uid": stable_uid("tess", wav),
            }
        )
    logger.info(f"[索引] TESS 样本: {len(out)}")
    return out


def index_asvp_esd(data_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    meta_lookup = load_metadata_lookup(data_dir, logger)
    audio_files = list_audio_files(data_dir)
    for audio in audio_files:
        stem = audio.stem
        emo: Optional[str] = None
        emo_raw: Optional[str] = None

        parts = stem.split("-")
        if len(parts) >= 3 and parts[2].isdigit():
            emo_raw = ASVP_EMOTION_CODE.get(parts[2], parts[2])
            emo = normalize_emotion(emo_raw)

        if emo is None:
            m = find_meta_for_audio(meta_lookup, audio, data_dir)
            emo = normalize_emotion(m.get("emotion"))
            emo_raw = m.get("emotion", emo_raw)

        if emo is None:
            emo = infer_emotion_from_path(audio)
            emo_raw = emo_raw or emo

        if emo is None:
            continue

        m = find_meta_for_audio(meta_lookup, audio, data_dir)
        text = m.get("text", "")
        out.append(
            {
                "dataset": "asvp_esd",
                "src_path": str(audio.resolve()),
                "emotion": emo,
                "emotion_raw": emo_raw or "",
                "text": text,
                "speaker": "",
                "uid": stable_uid("asvp_esd", audio),
            }
        )
    logger.info(f"[索引] ASVP-ESD 样本: {len(out)}")
    return out


def index_emo_emilia(data_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    jsonl_file = data_dir / "Emo-Emilia-ALL.jsonl"
    if jsonl_file.exists():
        out: List[Dict[str, Any]] = []
        with jsonl_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                emo_raw = str(row.get("emotion", "")).strip()
                emo = normalize_emotion(emo_raw)
                if emo is None:
                    continue
                wav_path = resolve_emo_emilia_audio_path(data_dir, str(row.get("wav", "")))
                if wav_path is None:
                    # 退化到 index 字段
                    idx = str(row.get("index", "")).strip()
                    if idx:
                        cand = data_dir / "wav" / f"{idx}.wav"
                        wav_path = cand.resolve() if cand.exists() else None
                if wav_path is None or not wav_path.exists():
                    continue
                text = str(row.get("text", "")).strip()
                out.append(
                    {
                        "dataset": "emo_emilia",
                        "src_path": str(wav_path),
                        "emotion": emo,
                        "emotion_raw": emo_raw,
                        "text": text,
                        "speaker": "",
                        "uid": stable_uid("emo_emilia", wav_path),
                    }
                )
        logger.info(f"[索引] Emo-Emilia(jsonl) 样本: {len(out)}")
        return out

    out: List[Dict[str, Any]] = []
    meta_lookup = load_metadata_lookup(data_dir, logger)
    audio_files = list_audio_files(data_dir)

    for audio in audio_files:
        m = find_meta_for_audio(meta_lookup, audio, data_dir)
        emo = normalize_emotion(m.get("emotion"))
        emo_raw = m.get("emotion", "")
        if emo is None:
            emo = infer_emotion_from_path(audio)
            emo_raw = emo_raw or (emo or "")
        if emo is None:
            continue
        text = m.get("text", "")
        out.append(
            {
                "dataset": "emo_emilia",
                "src_path": str(audio.resolve()),
                "emotion": emo,
                "emotion_raw": emo_raw,
                "text": text,
                "speaker": "",
                "uid": stable_uid("emo_emilia", audio),
            }
        )
    logger.info(f"[索引] Emo-Emilia 样本: {len(out)}")
    return out


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def build_merged_dataset(
    index_rows: List[Dict[str, Any]],
    selected_datasets: List[str],
    out_root: Path,
    link_mode: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in index_rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    ds_emotion_sets: Dict[str, Set[str]] = {}
    for ds in selected_datasets:
        ds_rows = by_dataset.get(ds, [])
        emo_set = {r["emotion"] for r in ds_rows if r.get("emotion")}
        ds_emotion_sets[ds] = emo_set
        logger.info(f"[交集] {ds}: 情感={sorted(emo_set)} | 样本数={len(ds_rows)}")

    valid_sets = [s for s in ds_emotion_sets.values() if s]
    if not valid_sets:
        raise RuntimeError("未找到有效情感标签，无法计算交集。")

    intersection = set.intersection(*valid_sets)
    if not intersection:
        raise RuntimeError("情感交集为空。请检查数据下载和标签映射。")
    intersection_sorted = sorted(intersection)
    logger.info(f"[交集] 最终情感交集: {intersection_sorted}")

    filtered = [r for r in index_rows if r.get("emotion") in intersection]
    logger.info(f"[合并] 交集过滤后样本数: {len(filtered)}")

    audio_root = out_root / "audio"
    manifest_root = out_root / "manifests"
    subtitles_root = out_root / "subtitles"
    for emo in intersection_sorted:
        (audio_root / emo).mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, Any]] = []
    subtitle_rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {e: 0 for e in intersection_sorted}

    for r in filtered:
        src = Path(r["src_path"])
        if not src.exists():
            continue
        emo = r["emotion"]
        uid = r["uid"]
        dst = audio_root / emo / f"{uid}{src.suffix.lower()}"
        link_or_copy(src, dst, link_mode)

        item = {
            "uid": uid,
            "dataset": r["dataset"],
            "emotion": emo,
            "emotion_raw": r.get("emotion_raw", ""),
            "audio": str(dst.resolve()),
            "src_audio": str(src.resolve()),
            "text": r.get("text", ""),
            "speaker": r.get("speaker", ""),
        }
        merged_rows.append(item)
        counts[emo] += 1

        text = str(r.get("text", "")).strip()
        if text:
            subtitle_rows.append(
                {
                    "uid": uid,
                    "dataset": r["dataset"],
                    "emotion": emo,
                    "audio": str(dst.resolve()),
                    "text": text,
                }
            )

    write_jsonl(manifest_root / "all.jsonl", merged_rows)
    for emo in intersection_sorted:
        write_jsonl(manifest_root / f"{emo}.jsonl", [x for x in merged_rows if x["emotion"] == emo])

    write_jsonl(subtitles_root / "subtitles.jsonl", subtitle_rows)
    with (subtitles_root / "subtitles.tsv").open("w", encoding="utf-8") as f:
        f.write("uid\tdataset\temotion\taudio\ttext\n")
        for r in subtitle_rows:
            f.write(
                f"{r['uid']}\t{r['dataset']}\t{r['emotion']}\t{r['audio']}\t{r['text'].replace(chr(9), ' ')}\n"
            )

    summary = {
        "selected_datasets": selected_datasets,
        "intersection_emotions": intersection_sorted,
        "num_rows_total_index": len(index_rows),
        "num_rows_after_intersection": len(merged_rows),
        "num_subtitles": len(subtitle_rows),
        "counts_by_emotion": counts,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[合并] 已写出 summary: {out_root / 'summary.json'}")
    return summary


def download_stage(
    datasets_root: Path,
    datasets: List[str],
    hf_token: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for ds in datasets:
        if ds == "ravdess":
            out[ds] = download_ravdess(datasets_root, logger)
        elif ds == "cremad":
            out[ds] = download_cremad(datasets_root, logger)
        elif ds == "tess":
            out[ds] = download_tess(datasets_root, logger)
        elif ds == "asvp_esd":
            out[ds] = download_asvp_esd(datasets_root, logger)
        elif ds == "emo_emilia":
            out[ds] = download_emo_emilia(datasets_root, logger, hf_token=hf_token)
        else:
            raise ValueError(f"不支持的数据集: {ds}")
    return out


def index_stage(
    datasets_root: Path,
    datasets: List[str],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        data_dir = datasets_root / ds / "data"
        if not data_dir.exists():
            logger.warning(f"[索引] 数据目录不存在，跳过 {ds}: {data_dir}")
            continue
        if ds == "ravdess":
            rows = index_ravdess(data_dir, logger)
        elif ds == "cremad":
            rows = index_cremad(data_dir, logger)
        elif ds == "tess":
            rows = index_tess(data_dir, logger)
        elif ds == "asvp_esd":
            rows = index_asvp_esd(data_dir, logger)
        elif ds == "emo_emilia":
            rows = index_emo_emilia(data_dir, logger)
        else:
            rows = []
        all_rows.extend(rows)
    return all_rows


def filter_stage(
    rows: List[Dict[str, Any]],
    apply_quality_filter: bool,
    apply_ser_filter: bool,
    min_duration_sec: float,
    max_duration_sec: float,
    max_silence_ratio: float,
    min_snr_db: float,
    quality_top_db: float,
    quality_cache_file: Path,
    ser_model_name: str,
    ser_hub: str,
    ser_min_conf: float,
    ser_cache_file: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cur = rows
    report: Dict[str, Any] = {
        "input_rows": len(rows),
        "quality_filter_enabled": bool(apply_quality_filter),
        "ser_filter_enabled": bool(apply_ser_filter),
    }

    if apply_quality_filter:
        cur, qstats = run_quality_filter(
            cur,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            max_silence_ratio=max_silence_ratio,
            min_snr_db=min_snr_db,
            top_db=quality_top_db,
            quality_cache_file=quality_cache_file,
            logger=logger,
        )
        report["quality"] = qstats

    if apply_ser_filter:
        cur, sstats = run_ser_filter(
            cur,
            ser_model_name=ser_model_name,
            ser_hub=ser_hub,
            ser_min_conf=ser_min_conf,
            ser_cache_file=ser_cache_file,
            logger=logger,
        )
        report["ser"] = sstats

    report["output_rows"] = len(cur)
    logger.info(f"[筛选] 完成: input={report['input_rows']} -> output={report['output_rows']}")
    return cur, report


def parse_stages(s: str) -> List[str]:
    stages = [x.strip().lower() for x in s.split(",") if x.strip()]
    valid = {"download", "index", "filter", "merge", "all"}
    bad = [x for x in stages if x not in valid]
    if bad:
        raise ValueError(f"非法 stages: {bad}")
    if "all" in stages:
        return ["download", "index", "filter", "merge"]
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载可公开获取情感语料，并构建“情感交集”大数据集（含字幕聚合）。"
    )
    parser.add_argument("--stages", type=str, default="all", help="逗号分隔: download,index,filter,merge,all")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ravdess,cremad,tess,emo_emilia",
        help="逗号分隔: ravdess,cremad,tess,emo_emilia,asvp_esd",
    )
    parser.add_argument("--datasets_root", type=Path, default=Path("./datasets_raw"))
    parser.add_argument("--output_root", type=Path, default=Path("./datasets_merged_intersection"))
    parser.add_argument("--index_file", type=Path, default=Path("./datasets_merged_intersection/index_all.jsonl"))
    parser.add_argument("--filtered_index_file", type=Path, default=Path("./datasets_merged_intersection/index_filtered.jsonl"))
    parser.add_argument("--link_mode", type=str, choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None))

    parser.add_argument("--disable_quality_filter", action="store_true", help="关闭附录B中的时长/静音/SNR筛选")
    parser.add_argument("--min_duration_sec", type=float, default=2.0)
    parser.add_argument("--max_duration_sec", type=float, default=20.0)
    parser.add_argument("--max_silence_ratio", type=float, default=0.30)
    parser.add_argument("--min_snr_db", type=float, default=10.0)
    parser.add_argument("--quality_top_db", type=float, default=40.0, help="静音判定阈值(dB)，默认40")
    parser.add_argument("--quality_cache_file", type=Path, default=Path("./datasets_merged_intersection/quality_cache.jsonl"))

    parser.add_argument("--disable_ser_filter", action="store_true", help="关闭附录B中的 emotion2Vec 置信度筛选")
    parser.add_argument("--ser_model", type=str, default="iic/emotion2vec_plus_large")
    parser.add_argument("--ser_hub", type=str, default="ms")
    parser.add_argument("--ser_min_conf", type=float, default=0.6)
    parser.add_argument("--ser_cache_file", type=Path, default=Path("./datasets_merged_intersection/ser_cache.jsonl"))

    parser.add_argument("--log_file", type=Path, default=None)
    args = parser.parse_args()

    if args.log_file is None:
        args.output_root.mkdir(parents=True, exist_ok=True)
        args.log_file = args.output_root / "build.log"
    logger = setup_logger(args.log_file)

    datasets = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    for ds in datasets:
        if ds not in DATASET_CHOICES:
            raise ValueError(f"不支持的数据集: {ds}")

    stages = parse_stages(args.stages)
    logger.info("========== 开始构建情感交集语料 ==========")
    logger.info(f"[参数] stages={stages} | datasets={datasets} | link_mode={args.link_mode}")
    logger.info(
        f"[参数-附录B] quality_filter={not args.disable_quality_filter} "
        f"(dur:[{args.min_duration_sec},{args.max_duration_sec}], "
        f"silence<={args.max_silence_ratio}, snr>={args.min_snr_db}) | "
        f"ser_filter={not args.disable_ser_filter} (min_conf={args.ser_min_conf})"
    )
    logger.info(f"[路径] datasets_root={args.datasets_root.resolve()}")
    logger.info(f"[路径] output_root={args.output_root.resolve()}")

    if "download" in stages:
        download_stage(
            datasets_root=args.datasets_root.resolve(),
            datasets=datasets,
            hf_token=args.hf_token,
            logger=logger,
        )

    rows: List[Dict[str, Any]]
    if "index" in stages:
        rows = index_stage(args.datasets_root.resolve(), datasets, logger)
        args.index_file.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.index_file, rows)
        logger.info(f"[索引] 已写出: {args.index_file.resolve()} | 条目={len(rows)}")
    else:
        rows = read_jsonl(args.index_file)
        logger.info(f"[索引] 从现有 index 读取: {args.index_file.resolve()} | 条目={len(rows)}")

    filtered_rows = rows
    if "filter" in stages:
        filtered_rows, report = filter_stage(
            rows=rows,
            apply_quality_filter=not bool(args.disable_quality_filter),
            apply_ser_filter=not bool(args.disable_ser_filter),
            min_duration_sec=float(args.min_duration_sec),
            max_duration_sec=float(args.max_duration_sec),
            max_silence_ratio=float(args.max_silence_ratio),
            min_snr_db=float(args.min_snr_db),
            quality_top_db=float(args.quality_top_db),
            quality_cache_file=args.quality_cache_file.resolve(),
            ser_model_name=str(args.ser_model),
            ser_hub=str(args.ser_hub),
            ser_min_conf=float(args.ser_min_conf),
            ser_cache_file=args.ser_cache_file.resolve(),
            logger=logger,
        )
        args.filtered_index_file.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.filtered_index_file, filtered_rows)
        logger.info(f"[筛选] 已写出 filtered index: {args.filtered_index_file.resolve()} | 条目={len(filtered_rows)}")
        (args.output_root.resolve() / "filter_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif args.filtered_index_file.exists():
        filtered_rows = read_jsonl(args.filtered_index_file)
        logger.info(f"[筛选] 使用已有 filtered index: {args.filtered_index_file.resolve()} | 条目={len(filtered_rows)}")
    elif "merge" in stages and (not args.disable_quality_filter or not args.disable_ser_filter):
        logger.info("[筛选] 未显式执行 filter 阶段，merge 前自动执行附录B筛选。")
        filtered_rows, report = filter_stage(
            rows=rows,
            apply_quality_filter=not bool(args.disable_quality_filter),
            apply_ser_filter=not bool(args.disable_ser_filter),
            min_duration_sec=float(args.min_duration_sec),
            max_duration_sec=float(args.max_duration_sec),
            max_silence_ratio=float(args.max_silence_ratio),
            min_snr_db=float(args.min_snr_db),
            quality_top_db=float(args.quality_top_db),
            quality_cache_file=args.quality_cache_file.resolve(),
            ser_model_name=str(args.ser_model),
            ser_hub=str(args.ser_hub),
            ser_min_conf=float(args.ser_min_conf),
            ser_cache_file=args.ser_cache_file.resolve(),
            logger=logger,
        )
        args.filtered_index_file.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.filtered_index_file, filtered_rows)
        logger.info(f"[筛选] 自动筛选后写出: {args.filtered_index_file.resolve()} | 条目={len(filtered_rows)}")
        (args.output_root.resolve() / "filter_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if "merge" in stages:
        summary = build_merged_dataset(
            index_rows=filtered_rows,
            selected_datasets=datasets,
            out_root=args.output_root.resolve(),
            link_mode=args.link_mode,
            logger=logger,
        )
        logger.info(f"[完成] 交集情感={summary['intersection_emotions']}")
        logger.info(f"[完成] 样本数={summary['num_rows_after_intersection']} | 字幕条目={summary['num_subtitles']}")

    logger.info("========== 全部流程结束 ==========")


if __name__ == "__main__":
    main()
