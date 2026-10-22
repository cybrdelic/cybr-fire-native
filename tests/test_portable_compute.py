"""Real compute dispatch/readback and teardown on the selected backend."""
from pathlib import Path
import sys
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cybrfire.gpu import ComputeContext

SOURCE = """#version 430
layout(local_size_x=128) in;
layout(std430,binding=0) buffer Values { float values[]; };
void main() { uint i=gl_GlobalInvocationID.x; if(i<257u) values[i]=2.0*float(i)+1.0; }
"""
for iteration in range(2):
    ctx = ComputeContext()
    try:
        with tempfile.TemporaryDirectory() as directory:
            shader = Path(directory) / "probe.comp"
            shader.write_text(SOURCE)
            buffer = ctx.buffer((257,))
            ctx.program(shader).run((257,), [buffer])
            np.testing.assert_array_equal(buffer.read(), 2*np.arange(257, dtype=np.float32)+1)
            ctx.check()
            print("PASS real compute/readback", iteration, ctx.info, flush=True)
    finally:
        ctx.close()
        ctx.close()
