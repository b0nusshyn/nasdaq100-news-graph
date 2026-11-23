"""Update individual historical metric leaders without promoting on a single score."""
import json
from pathlib import Path

from experiments.gat_regularization_v1.study import ROOT
from experiments.gat_regularization_v1.verify import read
from experiments.market_direction_v1.study import finite
from src.nlp.rebuild_timed_news import file_hash


def main():
    prior = ROOT.parent / "persistent_graph_v1/best_model_registry.json"
    old = json.loads(prior.read_text())
    scope = [ROOT, ROOT.parent / "gat_checkpoint_robust_v1"]
    new = []
    for study in scope:
        folder = study / "runs"
        for r in read(folder / "ledger.jsonl").values():
            new.append(dict(study=study.name, id=r["id"], seed=r["seed"], horizon=r["horizon"], config=r["config"],
                            dates=r["ranges"], metrics=r["assessment"], files=r["files"],
                            inputs_manifest=str(folder / "manifest.json"),
                            inputs_manifest_sha256=file_hash(folder / "manifest.json")))
    leads = old["individual_leaders"]
    field = {
        "accuracy": lambda m: m["accuracy"],
        "within_day_auc": lambda m: m["within_day_auc"],
        "mcc": lambda m: m["mcc"] if m["mcc"] is not None else -2,
        "accuracy_lift": lambda m: m["accuracy"] - m["majority"],
    }
    for h, group in leads.items():
        for k, fn in field.items():
            group[k] = max([group[k], *[r for r in new if r["horizon"] == int(h)]], key=lambda r: fn(r["metrics"]))
    fallback = Path("experiments/historical_demo_v1/development_replay.json")
    for p, h in json.loads(fallback.read_text())["sha256"].items():
        assert file_hash(Path(p)) == h
    (ROOT / "best_model_registry.json").write_text(json.dumps(finite(dict(prior=str(prior), leaders=leads,
        verified_fallback=str(fallback), cross_period_promotion=None, new_runs=new)), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
