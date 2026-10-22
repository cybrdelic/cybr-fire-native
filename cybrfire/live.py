"""Loopback-only control surface for the same solver used by offline capture.

The HTTP server is single-threaded because EGL context ownership is thread-local.
The renderer is not a second implementation and never owns simulation state.
"""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import io,json,time
from PIL import Image
from .gpu import ComputeContext
from .config import example
from .solver import Simulation
from .render import VolumeRenderer

class LiveEngine:
    def __init__(self,quality):self.ctx=None;self.reset('burner',quality,'auto');self.last_ms=0.
    def reset(self,name,quality,pressure):
        if pressure not in ['auto','multigrid','spectral']:raise ValueError('Invalid pressure backend')
        scene=example(name,quality);scene.validate()
        if self.ctx:self.ctx.close()
        self.ctx=ComputeContext()
        if pressure=='auto':pressure='spectral' if self.ctx.info['software_adapter'] else 'multigrid'
        scene=replace(scene,domain=replace(scene.domain,pressure_backend=pressure),render=replace(scene.render,width=480,height=640,samples=1,ground_resolution=48,ground_directions=16))
        self.sim=Simulation(self.ctx,scene);self.renderer=VolumeRenderer(self.ctx,self.sim);self.last_ms=0.;self.last_render_stats={}
    def frame(self,ticks=0,exposure=None,options=None):
        if not isinstance(ticks,int) or not 0<=ticks<=6:raise ValueError('ticks must be an integer from 0 to 6')
        cfg=self.renderer.cfg
        if exposure is not None:
            exposure=float(exposure)
            if not .02<=exposure<=4:raise ValueError('Exposure outside [0.02,4]')
            cfg=replace(cfg,exposure=exposure)
        if options is not None:
            allowed={'yaw','camera_distance','camera_height','target_height','illumination_directions','scattering_iterations','empty_space_skipping','stable_sampling','key_intensity','phase_g'}
            if not isinstance(options,dict) or set(options)-allowed:raise ValueError('Unknown render control')
            cfg=replace(cfg,**options)
            replace(self.sim.scene,render=cfg).validate()
        replace(self.sim.scene,render=cfg).validate()
        self.renderer.cfg=cfg
        start=time.perf_counter();self.sim.step(ticks);rgb=self.renderer.render(seed=self.sim.tick)
        self.last_render_stats=self.renderer.statistics()
        self.last_ms=(time.perf_counter()-start)*1000
        out=io.BytesIO();Image.fromarray(rgb).save(out,format='PNG');return out.getvalue()
    def status(self):return {'adapter':self.ctx.info,'scene':self.sim.scene.name,'grid':self.sim.shape,'time':self.sim.time,'tick':self.sim.tick,'pressure':'masked-pcg' if self.sim.scene.solids else self.sim.scene.domain.pressure_backend,'solids':len(self.sim.scene.solids),'particles':self.sim.embers.diagnostics() if self.sim.embers else None,'last_frame_ms':self.last_ms,'exposure':self.renderer.cfg.exposure,'render_statistics':self.last_render_stats,'render_options':{k:getattr(self.renderer.cfg,k) for k in ['yaw','camera_distance','illumination_directions','scattering_iterations','empty_space_skipping','stable_sampling','key_intensity','phase_g']}}

def serve(port=8765,quality='preview'):
    if not 1024<=port<=65535:raise ValueError('Port must be in [1024,65535]')
    engine=LiveEngine(quality);root=Path(__file__).resolve().parents[1]/'ui'
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,kind,status=200):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def json(self,obj,status=200):self.send(json.dumps(obj).encode(),'application/json',status)
        def do_GET(self):
            if self.path=='/api/status':return self.json(engine.status())
            routes={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
            if self.path not in routes:return self.json({'error':'Not found'},404)
            file,mime=routes[self.path];self.send((root/file).read_bytes(),mime)
        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and origin not in [f'http://127.0.0.1:{port}',f'http://localhost:{port}']:
                return self.json({'error':'Cross-origin state changes are disabled'},403)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('Expected a JSON object')
                if self.path=='/api/frame':return self.send(engine.frame(body.get('ticks',0),body.get('exposure'),body.get('render_options')),'image/png')
                if self.path=='/api/reset':
                    engine.reset(body.get('scene','burner'),body.get('quality','preview'),body.get('pressure','auto'));return self.json(engine.status())
                return self.json({'error':'Not found'},404)
            except (ValueError,TypeError,KeyError,json.JSONDecodeError) as e:self.json({'error':str(e)},400)
            except Exception as e:self.json({'error':str(e)},500)
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',port),Handler)
    print(f'CYBR FIRE 6 — http://127.0.0.1:{port} — {engine.ctx.info["renderer"]}',flush=True)
    try:server.serve_forever()
    finally:server.server_close();engine.ctx.close()
