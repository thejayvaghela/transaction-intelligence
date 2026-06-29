"""Distill the teacher into a tiny student; evaluate on test + gold (see docs/05-distillation.md).

  python scripts/distill.py            # distilled student -> models/student
The from-scratch ablation is just the plain trainer on the same student arch:
  python scripts/train_transformer.py --model prajjwal1/bert-tiny --output models/student-scratch
"""

import argparse
from pathlib import Path

import yaml

from transaction_intelligence import tracking
from transaction_intelligence.data.datasets import load_gold, load_split
from transaction_intelligence.eval import evaluate
from transaction_intelligence.models.distill import train_student
from transaction_intelligence.models.finetune import REPO_ROOT, TransformerPredictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to verify the pipeline")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "distill.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    teacher_dir = REPO_ROOT / cfg.get("teacher_dir", "models/teacher")
    print(f"loading teacher from {teacher_dir} ...")
    teacher = TransformerPredictor.load(teacher_dir, max_length=cfg.get("max_length", 48))

    train, val = load_split("train"), load_split("val")
    if args.smoke:
        train = train.sample(128, random_state=0)
        val = val.sample(64, random_state=0)

    student = train_student(train, val, teacher, cfg, smoke=args.smoke)
    out = student.save(REPO_ROOT / cfg.get("output_dir", "models/student"))
    print(f"saved -> {out}  (temperature={student.temperature:.3f})\n")

    tracking.setup()
    tag = Path(cfg.get("output_dir", "models/student")).name
    params = {
        "student_id": cfg["student_id"],
        "temperature": cfg["temperature"],
        "alpha": cfg["alpha"],
        "smoke": args.smoke,
    }
    with tracking.run(params=params, run_name=f"{tag}-smoke" if args.smoke else tag):
        for name, df in [("test", load_split("test")), ("gold", load_gold())]:
            res = evaluate(student, df, f"{tag}/{name}")
            print(res.summary())
            tracking.log_eval(res, name)


if __name__ == "__main__":
    main()
