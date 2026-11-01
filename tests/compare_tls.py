"""Compare the speed and results of GTLS, TLS 1.x (last release) and TLS 2.x.

Examples::

    python tests/compare_tls.py                       # standard suite, gtls/tls1/tls2
    python tests/compare_tls.py --suite full --plot
    python tests/compare_tls.py --scenarios tess_2min_1s --engines gtls,tls2,tls2-fused
    python tests/compare_tls.py --list

TLS releases are installed on first use into tests/.tls_envs (one directory per version),
results go to tests/.tls_compare_results/<timestamp>/ (report.md, report.json, plots).
"""
import argparse
import os
import sys

from tls_compare import (DEFAULT_ENGINES, SCENARIOS, SUITES, TLS1_VERSION, TLS2_VERSION,
                         plot_periodograms, registry, render_markdown, run_comparison)


def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", choices=sorted(SUITES), default="standard",
                        help="predefined scenario set (default: standard)")
    parser.add_argument("--scenarios", type=_csv, help="comma-separated scenarios (overrides --suite)")
    parser.add_argument("--engines", type=_csv, default=list(DEFAULT_ENGINES),
                        help="comma-separated engines (default: %s)" % ",".join(DEFAULT_ENGINES))
    parser.add_argument("--repeats", type=int, default=2,
                        help="warm runs after the cold run (default: 2)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="TLS use_threads (default: all %d cores)" % os.cpu_count())
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GTLS_GPU_DEVICE", 0)),
                        help="GTLS GPU device (default: $GTLS_GPU_DEVICE or 0)")
    parser.add_argument("--gpus", type=lambda v: [int(g) for g in _csv(v)],
                        help="split the GTLS search across these GPUs, e.g. 0,1")
    parser.add_argument("--tls1-version", default=TLS1_VERSION)
    parser.add_argument("--tls2-version", default=TLS2_VERSION)
    parser.add_argument("--baseline", default="tls1", help="engine used for speedups (default: tls1)")
    parser.add_argument("--timeout", type=float, help="seconds per engine and scenario")
    parser.add_argument("--output", help="output directory")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip filling the JIT/kernel caches before timing")
    parser.add_argument("--plot", action="store_true", help="save periodogram plots")
    parser.add_argument("--list", action="store_true", help="list scenarios, suites and engines")
    args = parser.parse_args(argv)

    if args.list:
        print("Scenarios:")
        for name, s in SCENARIOS.items():
            print("  %-16s %s" % (name, s.description))
        print("Suites:")
        for name, names in SUITES.items():
            print("  %-16s %s" % (name, ", ".join(names)))
        print("Engines:")
        for name, e in registry(args.tls1_version, args.tls2_version).items():
            print("  %-16s %s %s" % (name, e.label, e.options or ""))
        return 0

    report = run_comparison(
        args.scenarios or SUITES[args.suite], args.engines, outdir=args.output,
        repeats=args.repeats, threads=args.threads, gpu=args.gpu, gpus=args.gpus,
        timeout=args.timeout, baseline=args.baseline, tls1_version=args.tls1_version,
        tls2_version=args.tls2_version, warmup=not args.no_warmup,
        log=lambda *a: print(*a, flush=True))
    if args.plot:
        plot_periodograms(report)
    print()
    print(render_markdown(report))
    print("Report written to %s" % report["outdir"])
    failed = sum(s["failed"] for s in report["summary"].values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
