"""Train the transformer teacher; evaluate on test+gold via the harness; log to MLflow.

  python scripts/train_transformer.py --smoke   # tiny CPU run to verify the pipeline
  python scripts/train_transformer.py           # full run (use on Colab GPU)
"""

import argparse
from pathlib import Path

import yaml

from transaction_intelligence import tracking
from transaction_intelligence.data.datasets import load_gold, load_split
from transaction_intelligence.eval import evaluate
from transaction_intelligence.models.finetune import REPO_ROOT, train_transformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to verify the pipeline")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "finetune.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    train, val = load_split("train"), load_split("val")
    if args.smoke:
        train = train.sample(128, random_state=0)
        val = val.sample(64, random_state=0)

    model = train_transformer(train, val, cfg, smoke=args.smoke)
    out = model.save(REPO_ROOT / cfg.get("output_dir", "models/teacher"))
    print(f"saved -> {out}  (temperature={model.temperature:.3f})\n")

    tracking.setup()
    params = {"model_id": cfg["model_id"], "smoke": args.smoke}
    for k in ("learning_rate", "batch_size", "epochs", "max_length", "seed"):
        if k in cfg:
            params[k] = cfg[k]
    with tracking.run(params=params, run_name="teacher-smoke" if args.smoke else "teacher"):
        for name, df in [("test", load_split("test")), ("gold", load_gold())]:
            res = evaluate(model, df, f"teacher/{name}")
            print(res.summary())
            tracking.log_eval(res, name)


if __name__ == "__main__":
    main()
