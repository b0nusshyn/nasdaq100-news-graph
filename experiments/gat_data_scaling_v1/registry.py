"""Register both history experiments without promoting a retrospective maximum."""
import json

from experiments.gat_data_scaling_v1.study import ROOT, records
from experiments.market_direction_v1.study import finite
from src.nlp.rebuild_timed_news import file_hash


def main():
    prior = ROOT.parent / "gat_joint_v1/best_model_registry.json"
    leaders = json.loads(prior.read_text())["leaders"]
    new = []
    for d in ("runs", "recency_runs"):
        folder = ROOT / d
        for r in records(folder/"ledger.jsonl").values():
            new.append(dict(id=r["id"], seed=r["seed"], horizon=r["horizon"], ranges=r["ranges"],
                            config=r["config"], metrics=r["assessment"], files=r["files"],
                            inputs_manifest=str(folder/"manifest.json"),
                            inputs_manifest_sha256=file_hash(folder/"manifest.json")))
    assert len(new) == 20
    selectors = {"accuracy": lambda m: m["accuracy"], "within_day_auc": lambda m: m["within_day_auc"],
                 "mcc": lambda m: m["mcc"] if m["mcc"] is not None else -1,
                 "accuracy_lift": lambda m: m["accuracy"]-m["majority"]}
    for k, fn in selectors.items():
        leaders["5"][k] = max([leaders["5"][k], *new], key=lambda r: fn(r["metrics"]))
    data = dict(source=str(prior), leaders=leaders, new_runs=new, independent_test=False,
                promoted=None, fallback=str(ROOT.parent/"historical_demo_v1/development_replay.json"))
    (ROOT/"best_model_registry.json").write_text(json.dumps(finite(data), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
