"""Report actual host/runtime capabilities; never substitutes media for compute."""
import argparse
import importlib.metadata
import json
import platform
import shutil

def diagnose(probe=True):
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "dependencies": {}, "ffmpeg": bool(shutil.which("ffmpeg")),
              "ffprobe": bool(shutil.which("ffprobe")),
              "docker": bool(shutil.which("docker")),
              "compute_supported": None, "adapter": None}
    for name in ("numpy", "scipy", "Pillow", "glfw"):
        try:
            report["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][name] = None
    if probe:
        try:
            from .gpu import ComputeContext
            ctx = ComputeContext()
            try:
                report.update(compute_supported=True, adapter=ctx.info)
            finally:
                ctx.close()
        except (RuntimeError, ImportError, OSError, ValueError) as exc:
            report.update(compute_supported=False, reason=str(exc))
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-compute", action="store_true")
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args()
    if args.no_probe and args.require_compute:
        parser.error("--require-compute cannot be combined with --no-probe")
    report = diagnose(not args.no_probe)
    print(json.dumps(report, indent=2))
    return 2 if args.require_compute and report["compute_supported"] is not True else 0

if __name__ == "__main__":
    raise SystemExit(main())
