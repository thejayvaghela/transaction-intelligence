"""Train the baseline, evaluate on test + gold, compare to the rules floor (`make baseline`)."""

import json

from transaction_intelligence.data.datasets import load_gold, load_split
from transaction_intelligence.eval import evaluate
from transaction_intelligence.models.baseline import train_baseline
from transaction_intelligence.models.rules import RulesPredictor


def main() -> None:
    train = load_split("train")
    print(f"training baseline on {len(train)} rows ...")
    model = train_baseline(train)
    path = model.save()
    print(f"saved -> {path}\n")

    rules = RulesPredictor()
    results = {}
    for name, df in [("test", load_split("test")), ("gold", load_gold())]:
        b = evaluate(model, df, f"baseline/{name}")
        r = evaluate(rules, df, f"rules/{name}")
        print(b.summary())
        print(r.summary())
        print()
        results[name] = {"baseline": b.to_dict(), "rules": r.to_dict()}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
