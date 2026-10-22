"""Run the unchanged FIRE engine with Mesa software compute in existing Docker.

Does not install Docker, start a daemon, mount user folders or grant GPU access.
The source is copied at build time; use --build after source changes.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "cybr-fire-native:portable-local"

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build the local image from this checkout")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, help="Mount only this directory read/write at /output")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Python arguments after --, e.g. -- -m cybrfire device")
    args = parser.parse_args(argv)
    docker = shutil.which("docker")
    if not docker:
        parser.exit(2, "Docker executable not found. Install/start Docker separately, or use a native OpenGL 4.5 environment. No system changes were made.\n")
    try:
        check = subprocess.run([docker, "info", "--format", "{{.OSType}}"],
                               capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"Cannot contact the existing Docker daemon: {exc}. Start it separately.\n")
    if check.returncode or check.stdout.strip() != "linux":
        parser.exit(2, "A running Linux-container Docker daemon is required. Start/switch Docker separately.\n" + check.stderr[-1000:])
    if args.build:
        result = subprocess.run([docker, "build", "--tag", args.image, str(ROOT)])
        if result.returncode:
            return result.returncode
    else:
        image = subprocess.run([docker, "image", "inspect", args.image], capture_output=True)
        if image.returncode:
            parser.exit(2, "Local image not found. Run this command with --build first.\n")
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not command:
        command = ["-m", "cybrfire", "device"]
    run = [docker, "run", "--rm", "--network=none", "--entrypoint", "python"]
    if args.output:
        output = args.output.expanduser().resolve()
        if "," in str(output):
            parser.error("--output cannot contain a comma (Docker mount syntax)")
        output.mkdir(parents=True, exist_ok=True)
        run += ["--mount", f"type=bind,source={output},target=/output"]
    run += [args.image, *command]
    print("Running the unchanged solver using Linux Mesa software compute; this is not a native macOS GPU backend.", flush=True)
    try:
        return subprocess.run(run).returncode
    except KeyboardInterrupt:
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
