"""Evolve V7's MAC solver; publish world-space, immutable velocity snapshots.

Input may be the previously supplied V7 complete ZIP or its start checkpoint.
Changing the integration tick is an explicit initial-condition import, not an
identical restart. Run this BEFORE allocating the large fine-detail grid.
"""
from pathlib import Path
from dataclasses import replace
import argparse, hashlib, json, shutil, sys, time, zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cybrfire.config import scene_from_dict
from cybrfire.gpu import ComputeContext
from cybrfire.solver import Simulation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='V7 complete ZIP or v7-start.npz')
    parser.add_argument('--member', default='CYBR_FIRE_V7/cache/v7-start.npz')
    parser.add_argument('--out', type=Path, default=Path('cache/parent'))
    parser.add_argument('--snapshots', type=int, default=33)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--tick-rate', type=int, default=120)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.snapshots < 2 or args.fps < 1 or args.tick_rate % args.fps:
        raise ValueError('Need >=2 snapshots and an integral number of ticks/frame')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Choose a new cache directory; existing states are not overwritten')
    start = out / 'start.npz'
    if args.input.suffix.lower() == '.zip':
        with zipfile.ZipFile(args.input) as archive:
            with archive.open(args.member) as source, start.open('wb') as destination:
                shutil.copyfileobj(source, destination)
    elif args.input.resolve() != start.resolve():
        shutil.copyfile(args.input, start)
    ctx = ComputeContext()
    sim = None
    try:
        with np.load(start, allow_pickle=False) as state:
            original = scene_from_dict(json.loads(str(state['config'])))
            start_time = int(state['tick']) * original.domain.tick
            scene = replace(original, domain=replace(original.domain,
                native_fields=True, tick=1 / args.tick_rate,
                trace_courant=1., max_trace_steps=16))
            sim = Simulation(ctx, scene)
            sim.q.write(state['q'])
            sim.vel.write(state['velocity'])
            sim.tick = round(start_time / scene.domain.tick)
            if abs(sim.time - start_time) > 1e-9:
                raise ValueError('Checkpoint time is not aligned to the requested tick')
            np.save(out / 'initial-q.npy', state['q'])
        manifest = dict(scene=scene.to_dict(), start_time=start_time,
            frame_dt=1 / args.fps, grid=sim.shape, adapter=ctx.info,
            input_sha256=digest(start), frames=[], complete=False,
            import_is_equivalent_restart=False)
        publish(out / 'manifest.json', manifest)
        for index in range(args.snapshots):
            began = time.perf_counter()
            if index:
                sim.step(args.tick_rate // args.fps)
            ctx.finish()
            dims = tuple(n + 1 for n in sim.shape)
            destination = np.lib.format.open_memmap(
                out / f'velocity-{index:03}.npy', mode='w+',
                dtype=np.float32, shape=sim.vel.shape)
            for z in range(0, dims[2], 8):
                slab = sim.vel.read_slab(z, min(8, dims[2] - z))
                offset = z * dims[0] * dims[1]
                destination[offset:offset + slab.size // 4] = slab.reshape(-1, 4)
            destination.flush()
            del destination
            diagnostics = sim.diagnostics() if index % 8 == 0 else {'time': sim.time}
            manifest['frames'].append(dict(index=index, time=sim.time,
                seconds=time.perf_counter() - began, diagnostics=diagnostics))
            publish(out / 'manifest.json', manifest)
            print(index, sim.time, manifest['frames'][-1]['seconds'], flush=True)
        manifest['complete'] = True
        publish(out / 'manifest.json', manifest)
    finally:
        ctx.close()


if __name__ == '__main__':
    main()
