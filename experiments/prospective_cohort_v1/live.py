"""Local, fail-closed prospective capture. No backdating or network acquisition."""
import argparse
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import sklearn

from experiments.boosted_decision_v1.replay import confidence_result, ranked, ranking_select
from experiments.market_direction_v1.study import features, finite
from src.models.ladder_report import auc, within_day_auc

ROOT = Path(__file__).parent
CAL = mcal.get_calendar("NASDAQ")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return pd.Timestamp(datetime.now(timezone.utc))


def timestamp(text):
    t = pd.Timestamp(text)
    if pd.isna(t) or t.tz is None:
        raise ValueError("timestamp must include an explicit timezone")
    return t.tz_convert("UTC")


def calendar(day):
    start = pd.Timestamp(day)
    if day != start.date().isoformat():
        raise ValueError("session must use YYYY-MM-DD format")
    sessions = CAL.schedule(start - pd.Timedelta(days=120), start + pd.Timedelta(days=14))
    if start not in sessions.index:
        raise ValueError("signal is not a NASDAQ session")
    return sessions, sessions.index.get_loc(start)


def timing(day, frozen_at, clock):
    sessions, i = calendar(day)
    close = sessions.market_close.iloc[i]
    next_open = sessions.market_open.iloc[i+1]
    if not (timestamp(frozen_at) < close <= clock < next_open):
        raise ValueError("signal must close after freeze and be captured before next-session open")
    return close, next_open, sessions.index[i+1].date().isoformat()


def write_once(path, data, *, deadline=None):
    """Publish complete bytes atomically and reject an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        os.chmod(tmp.name, 0o444)
        if deadline is not None and now() >= deadline:
            raise ValueError("next session opened before commit")
        os.link(tmp.name, path)
    return digest(data)


def encode(data):
    return (json.dumps(finite(data), sort_keys=True, allow_nan=False) + "\n").encode()


def load_freeze():
    raw = (ROOT / "freeze.json").read_bytes()
    cfg = json.loads(raw)
    deps = (sklearn, pd, np, mcal)
    if any(p.__version__ != cfg["dependencies"][p.__name__] for p in deps):
        raise ValueError("frozen dependency version mismatch")
    for key in ["experiments/market_direction_v1/study.py", "experiments/boosted_decision_v1/replay.py"]:
        if digest((ROOT.parents[1] / key).read_bytes()) != cfg["inputs_sha256"][key]:
            raise ValueError("frozen feature or decision implementation changed")
    return cfg, digest(raw)


def source_gate(path, cfg):
    gate = json.loads(path.read_bytes())
    if gate.get("status") != "approved" or not path.with_name("source_evidence.md").is_file():
        raise ValueError("live source has not passed the readiness gate")
    if digest(path.with_name("source_evidence.md").read_bytes()) != gate["evidence_sha256"]:
        raise ValueError("source audit evidence hash mismatch")
    ids = gate["security_ids"]
    if set(ids) != set(cfg["cohort"]) or not all(ids.values()) or len(set(ids.values())) != len(ids):
        raise ValueError("source gate requires stable distinct security identifiers")
    return gate


def snapshot(path, meta_path, gate, cfg, dates, earliest, clock):
    raw, meta_raw = path.read_bytes(), meta_path.read_bytes()
    meta = json.loads(meta_raw)
    for k in ("source", "adjustment_policy", "security_ids"):
        if meta[k] != gate[k]:
            raise ValueError(f"source mismatch for {k}")
    if not meta.get("revision") or digest(raw) != meta["sha256"]:
        raise ValueError("missing source revision or snapshot hash mismatch")
    if not earliest <= timestamp(meta["published_at"]) <= timestamp(meta["retrieved_at"]) <= clock:
        raise ValueError("snapshot availability timeline invalid")
    p = pd.read_parquet(io.BytesIO(raw))
    wanted = ["open", "high", "low", "close", "adj_close", "volume"]
    p = p[["date", "ticker", *wanted]].copy()
    p.date = pd.to_datetime(p.date)
    keys = pd.MultiIndex.from_frame(p[["date", "ticker"]])
    expected = pd.MultiIndex.from_product([dates, cfg["cohort"]], names=keys.names)
    if not keys.is_unique or len(keys) != len(expected) or not keys.isin(expected).all():
        raise ValueError("snapshot must cover exactly the requested cohort sessions")
    if (not np.isfinite(p[wanted]).all().all() or p[wanted[:-1]].le(0).any().any()
        or p.volume.le(0).any() or (p.high < p[["open", "close"]].max(axis=1)).any()
        or (p.low > p[["open", "close"]].min(axis=1)).any()):
        raise ValueError("invalid OHLCV or adjustment")
    return p, raw, meta_raw, meta


def preserve(log, raw, meta_raw):
    for payload in (raw, meta_raw):
        path = log / "inputs" / digest(payload)
        try:
            write_once(path, payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("stored input hash collision or mutation") from None


def issue(args, cfg, freeze_hash, gate):
    clock = now()
    close, deadline, end = timing(args.day, cfg["frozen_at"], clock)
    sessions, idx = calendar(args.day)
    dates = sessions.index[idx+1-cfg["warmup_sessions"]:idx+1]
    p, raw, meta_raw, meta = snapshot(args.bars, args.meta, gate, cfg, dates, close, clock)
    multiplier = p.adj_close / p.close
    for col in ("open", "high", "low", "close"):
        p[col] *= multiplier
    feat, _ = features(p.assign(in_index=True))
    latest = feat[feat.date.eq(pd.Timestamp(args.day))].copy()
    model_bytes = (ROOT / "model.joblib").read_bytes()
    if digest(model_bytes) != cfg["model_sha256"]:
        raise ValueError("frozen model hash changed")
    model = joblib.load(io.BytesIO(model_bytes))
    if (list(model.classes_) != [0, 1] or list(model.feature_names_in_) != cfg["feature_columns"]
        or any(model.get_params()[k] != v for k, v in cfg["model_params"].items())):
        raise ValueError("model schema mismatch")
    values = latest[cfg["feature_columns"]].astype(float)
    if len(latest) != len(cfg["cohort"]) or not np.isfinite(values).all().all():
        raise ValueError("required signal-day features missing")
    latest["score"] = model.decision_function(values)
    r = ranked(latest, cfg["calibration_prevalence"])
    for level in cfg["selection_levels"]:
        r[f"selected_{level}"] = r.index.isin(ranking_select(r, level).index)
    cols = ["ticker", "score", "rank_score", "rank_call", *[f"selected_{x}" for x in cfg["selection_levels"]]]
    preserve(args.log, raw, meta_raw)
    issued = now()
    if issued >= deadline:
        raise ValueError("next session opened before commit")
    record = dict(signal=args.day, target=end, issued_at=issued.isoformat(), snapshot_sha256=digest(raw),
                  metadata_sha256=digest(meta_raw), freeze_sha256=freeze_hash, model_sha256=cfg["model_sha256"],
                  source=meta["source"], revision=meta["revision"], gate_sha256=digest(args.gate.read_bytes()),
                  predictions=r[cols].to_dict("records"))
    path = args.log / "predictions" / f"{args.day}.json"
    print(write_once(path, encode(record), deadline=deadline), path)


def settle(args, cfg, freeze_hash, gate):
    path = args.log / "predictions" / f"{args.day}.json"
    raw_pred = path.read_bytes()
    pred = json.loads(raw_pred)
    issued = timestamp(pred["issued_at"])
    _, _, end = timing(args.day, cfg["frozen_at"], issued)
    if (pred["target"] != end or pred["signal"] != args.day or pred["freeze_sha256"] != freeze_hash
        or pred["gate_sha256"] != digest(args.gate.read_bytes())):
        raise ValueError("invalid prediction linkage")
    sessions, i = calendar(end)
    close = sessions.market_close.iloc[i]
    clock = now()
    if clock < close:
        raise ValueError("target session has not closed")
    dates = pd.to_datetime([args.day, end])
    p, raw, meta_raw, _ = snapshot(args.bars, args.meta, gate, cfg, dates, close, clock)
    wide = p.pivot(index="date", columns="ticker", values="adj_close")
    ret = wide.iloc[1] / wide.iloc[0] - 1
    excess = ret - ret.mean()
    rows = [{"ticker": k, "return": ret[k], "excess": excess[k], "y_true": int(excess[k] > 0)} for k in ret.index]
    record = dict(signal=args.day, target=end, settled_at=now().isoformat(), prediction_sha256=digest(raw_pred),
                  snapshot_sha256=digest(raw), metadata_sha256=digest(meta_raw), outcomes=rows)
    preserve(args.log, raw, meta_raw)
    path = args.log / "outcomes" / f"{args.day}.json"
    print(write_once(path, encode(record)), path)


def evaluate(log):
    frames = []
    cfg, freeze_hash = load_freeze()
    for p in sorted((log / "outcomes").glob("*.json")):
        outcome = json.loads(p.read_bytes())
        raw = (log / "predictions" / p.name).read_bytes()
        pred = json.loads(raw)
        if (digest(raw) != outcome["prediction_sha256"] or pred["signal"] != outcome["signal"]
            or pred["target"] != outcome["target"] or pred["freeze_sha256"] != freeze_hash
            or pred["model_sha256"] != cfg["model_sha256"] or p.stem != pred["signal"]):
            raise ValueError("outcome linkage invalid")
        _, _, end = timing(pred["signal"], cfg["frozen_at"], timestamp(pred["issued_at"]))
        sessions, idx = calendar(end)
        if end != outcome["target"] or timestamp(outcome["settled_at"]) < sessions.market_close.iloc[idx]:
            raise ValueError("outcome timing invalid")
        f = pd.DataFrame(pred["predictions"]).merge(pd.DataFrame(outcome["outcomes"]), on="ticker",
                                                      validate="one_to_one", how="outer", indicator=True)
        if not f._merge.eq("both").all() or set(f.ticker) != set(cfg["cohort"]):
            raise ValueError("outcome population mismatch")
        frames.append(f.assign(date=pred["signal"]))
    if not frames:
        return {"settled_sessions": 0}
    f = pd.concat(frames, ignore_index=True)
    scores = {}
    for col in (x for x in f if x.startswith("selected_")):
        included = f[f[col]]
        scores[col] = confidence_result(f, included, "rank_call")
        scores[col]["within_day_auc"] = within_day_auc(included.assign(p_up=included.score))
        if len(frames) < 10:
            for field in ("accuracy_interval", "excess_interval"):
                scores[col].pop(field, None)
        else:
            day_auc = pd.Series({day: auc(g.y_true, g.score) for day, g in included.groupby("date")})
            values = day_auc.reindex(f.date.drop_duplicates()).to_numpy()
            n = len(values)
            start = np.random.default_rng(42).integers(n, size=(2500, (n+9)//10, 1))
            draws = values[((start + np.arange(10)) % n).reshape(2500, -1)[:, :n]]
            valid = np.isfinite(draws).any(axis=1)
            if valid.any():
                scores[col]["auc_interval"] = np.quantile(np.nanmean(draws[valid], axis=1), [.025, .975]).tolist()
    return dict(settled_sessions=len(frames), metrics=scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["issue", "settle", "evaluate"])
    parser.add_argument("--log", type=Path, default=ROOT / "log")
    parser.add_argument("--gate", type=Path, default=ROOT / "source_status.json")
    parser.add_argument("--day")
    parser.add_argument("--bars", type=Path)
    parser.add_argument("--meta", type=Path)
    args = parser.parse_args()
    if args.action == "evaluate":
        print(encode(evaluate(args.log)).decode(), end="")
        return
    if not all((args.day, args.bars, args.meta)):
        parser.error("issue and settle require --day, --bars and --meta")
    cfg, freeze_hash = load_freeze()
    gate = source_gate(args.gate, cfg)
    action = issue if args.action == "issue" else settle
    action(args, cfg, freeze_hash, gate)


if __name__ == "__main__":
    main()
