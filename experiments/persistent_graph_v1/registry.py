"""Preserve individual metric leaders and the separate reproducible fallback."""
import json

from experiments.market_direction_v1.study import finite
from experiments.persistent_graph_v1.study import ROOT
from experiments.persistent_graph_v1.verify import ledger
from src.nlp.rebuild_timed_news import file_hash


def main():
    path = ROOT.parent / "submission_readiness_v1/best_model_registry.json"
    old = json.loads(path.read_text())
    new = ledger(ROOT / "runs/ledger.jsonl")
    artifacts = json.loads((ROOT / "artifact_hashes.json").read_text())
    source = ROOT / "runs/manifest.json"
    identities = [r for r in old["rows"] if r["dates"][1] <= "2023-12-31"]
    for name, r in new.items():
        identities.append(dict(model=r["arm"], period=r["fold"], seed=r["seed"], horizon=r["horizon"],
                               dates=r["ranges"]["assessment"], metrics=r["metrics"],
                               config=r["config"], inputs_manifest=str(source),
                               inputs_manifest_sha256=file_hash(source), **artifacts[name]))
    leaders = {}
    selectors = {
        "accuracy": lambda m: m["accuracy"],
        "within_day_auc": lambda m: m["within_day_auc"],
        "mcc": lambda m: m["mcc"] if m["mcc"] is not None else -2,
        "accuracy_lift": lambda m: m["accuracy"] - m["majority"],
    }
    for h in (1, 5):
        scope = [r for r in identities if r["horizon"] == h]
        leaders[h] = {k: max(scope, key=lambda r: fn(r["metrics"])) for k, fn in selectors.items()}
    fallback = ROOT.parent / "historical_demo_v1/development_replay.json"
    result = dict(prior_registry=str(path), new_records=len(new),
                  record_scope="persisted 2019–2023 direction fits only",
                  individual_leaders=leaders, demonstration_fallback=json.loads(fallback.read_text()),
                  reproducible_cross_period_winner=None)
    (ROOT / "best_model_registry.json").write_text(json.dumps(finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
