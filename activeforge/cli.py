"""ActiveForge CLI 入口。

用法：
  python -m activeforge.cli bench  [--out benchmark.json] [--learner rf]
  python -m activeforge.cli selftest
"""
from __future__ import annotations

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def _cmd_bench(args) -> int:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from activeforge.examples.run_demo import build_report

    report = build_report(learner_kind=args.learner, write_json=bool(args.out))
    if args.out:
        from pathlib import Path as _P

        _P(args.out).write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"benchmark.json 已写入: {args.out}")
    _print_summary(report)
    return 0


def _print_summary(report: dict) -> None:
    print("\n=== ActiveForge 基准摘要 ===")
    print(f"{'dataset':<12}{'strategy':<12}{'mean_acc':<10}{'std':<8}{'Δrandom':<9}{'sig':<5}")
    for r in report["results"]:
        print(f"{r['dataset']:<12}{r['strategy']:<12}{r['mean_acc']:<10}{r['std_acc']:<8}"
              f"{r['delta_acc_vs_random']:<9}{'Y' if r['significant_vs_random'] else '-':<5}")
    agg = report["agg_vs_random"]
    print(f"\n聚合(最佳主动 vs 随机): Δacc={agg['mean_delta_acc']}  Δauc={agg['mean_delta_auc']}  "
          f"显著数据集 {agg['n_significant']}/{agg['n_datasets']}")
    print(f"质量等级: {report['quality_grade']}  | 确定性一致: {report['determinism_check']['identical']}")


def _cmd_selftest(args) -> int:
    import subprocess

    py = sys.executable
    rc = subprocess.call([py, "-m", "pytest", "-q", "-W", "ignore::UserWarning",
                          str(args.tests)])
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="activeforge", description="ActiveForge 主动学习系统")
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("bench", help="运行基准并生成报告")
    b.add_argument("--out", default=None, help="benchmark.json 输出路径")
    b.add_argument("--learner", default="rf", choices=["rf", "lr", "xgb", "lgbm"])
    b.set_defaults(func=_cmd_bench)
    s = sub.add_parser("selftest", help="运行 pytest 单测")
    s.add_argument("--tests", default="tests", help="测试目录")
    s.set_defaults(func=_cmd_selftest)
    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
