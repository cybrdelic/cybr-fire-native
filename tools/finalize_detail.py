#!/usr/bin/env python3
"""Verify actual capture bytes, extract/test the source, and package V8.

This is deliberately separate from the numerical process: the latter releases
its multi-gigabyte fields before encoders and regression tests allocate memory.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from datetime import datetime, timezone
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent
OUT=DATA/'cybr-fire-v8-delivery'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def core_digest(root):
    h=hashlib.sha256()
    for path in sorted([*root.glob('cybrfire/*.py'),*root.glob('shaders/*')]):
        h.update(path.relative_to(root).as_posix().encode());h.update(path.read_bytes())
    return h.hexdigest()

def run(command,log=None,cwd=None):
    if log:
        with Path(log).open('w') as stream:
            result=subprocess.run(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
    else:result=subprocess.run(command,cwd=cwd)
    if result.returncode:raise RuntimeError(f'Command failed ({result.returncode}): {command}')

def archive_tree(root,path):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in sorted(root.rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts:
                archive.write(file,root.name+'/'+file.relative_to(root).as_posix())
    with zipfile.ZipFile(path) as archive:
        bad=archive.testzip()
        if bad:raise RuntimeError('Archive corruption: '+bad)
    return dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path),integrity='passed')

def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
        '-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration',
        '-of','json',str(path)]))['streams'][0]

def main():
    if not (OUT/'capture.json').is_file():raise RuntimeError('No capture metadata')
    capture=json.loads((OUT/'capture.json').read_text())
    if capture['status']!='completed_and_fully_decoded':raise RuntimeError('Capture not completed')
    video=OUT/'V8_detail_rebuild_4K.mp4';spec=probe(video)
    run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'])
    if (spec['width'],spec['height'],int(spec['nb_read_frames']))!=(2160,3840,24):
        raise AssertionError('Unexpected actual video dimensions/frame count')
    if spec['r_frame_rate']!='24/1':raise AssertionError('Wrong playback rate')
    if digest(video)!=capture['video_sha256']:raise AssertionError('Video hash mismatch')
    if core_digest(ROOT)!=capture['source_sha256']:raise AssertionError('Executed core was changed')
    if capture['fine_scalar_cells']!=99_090_432 or capture['parent_velocity_pressure_cells']!=7_168_000:
        raise AssertionError('Grid-count mismatch')
    state=capture['final_state']
    if not state['finite'] or min(state['minimum'])<0:raise AssertionError('Invalid final scalar state')
    if state['maximum'][1]>capture['scene']['chemistry']['oxygen_ambient']+2e-6:
        raise AssertionError('Invalid oxygen deficit')
    if state['cells']!=99_090_432 or state['hot_cells']==0:raise AssertionError('Final field mismatch')
    frames=capture['frames'];previous=None;differences=[];hashes=[]
    for i,frame in enumerate(frames):
        file=OUT/'frames'/f'{i:04}.png'
        if digest(file)!=frame['sha256']:raise AssertionError('Frame hash mismatch')
        hashes.append(frame['sha256'])
        with Image.open(file) as image:
            image.load()
            if image.size!=(2160,3840):raise AssertionError('Frame dimensions mismatch')
            reduced=np.array(image.resize((144,256)),dtype=np.float32)
        if previous is not None:differences.append(float(np.mean(abs(reduced-previous))))
        previous=reduced
        if i and abs(frame['time']-frames[i-1]['time']-1/24)>1e-10:
            raise AssertionError('Physical frame times are not 24 Hz')
        if i and frame['fine_steps_completed']-frames[i-1]['fine_steps_completed']!=2:
            raise AssertionError('Fine timestep accounting mismatch')
    if len(set(hashes))!=24 or max(differences)==0:raise AssertionError('Repeated/unchanging frames')
    for i in (0,12,23):
        if digest(OUT/f'V8_detail_still_{i:02}.png')!=frames[i]['sha256']:
            raise AssertionError('Lossless still is not its declared original frame')
    # These are explicitly 1:1 crops, not separately simulated or superscaled views.
    with Image.open(OUT/'frames/0012.png') as image:
        image.crop((480,1720,1680,3320)).save(OUT/'V8_native_closeup.png')
        image.crop((480,2000,1680,3200)).save(OUT/'V8_native_square_crop.png')
    detail_video=OUT/'V8_native_closeup.mp4'
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','24','-i',
        str(OUT/'frames/%04d.png'),'-vf','crop=1200:1600:480:1720',
        '-c:v','libx264','-preset','slow','-crf','12','-pix_fmt','yuv420p',
        '-movflags','+faststart','-threads','4',str(detail_video)])
    run(['ffmpeg','-v','error','-i',str(detail_video),'-f','null','-'])
    detail_spec=probe(detail_video)
    if (detail_spec['width'],detail_spec['height'],int(detail_spec['nb_read_frames']))!=(1200,1600,24):
        raise AssertionError('Invalid close-up encode')
    # Prepare a self-contained SOURCE archive, not a disguised image asset pack.
    staging=DATA/'v8-release-staging'/'CYBR_FIRE_V8'
    if staging.exists():shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for folder in ('cybrfire','shaders','tests'):
        shutil.copytree(ROOT/folder,staging/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (staging/'tools').mkdir();(staging/'docs').mkdir();(staging/'baseline').mkdir()
    for name in ('capture_detail.py','build_parent_cache.py','reproduce_v8.sh','finalize_detail.py'):
        shutil.copyfile(ROOT/'tools'/name,staging/'tools'/name)
    for name in ('requirements.txt','requirements-tested.txt'):
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,staging/name)
    shutil.copyfile(ROOT/'README_V8.md',staging/'README.md')
    shutil.copyfile(ROOT/'docs/DETAIL_REBUILD.md',staging/'docs/DETAIL_REBUILD.md')
    shutil.copyfile(ROOT/'README.md',staging/'baseline/README_V7.md')
    shutil.copyfile(ROOT/'baseline/V7_source.zip',staging/'baseline/V7_source.zip')
    shutil.copyfile(ROOT/'baseline/executed_parent_cache_builder.py',staging/'baseline/executed_parent_cache_builder.py')
    for folder in ('evidence','evidence-v8'):
        (staging/folder).mkdir();(staging/folder/'.gitkeep').write_text('')
    source=DATA/'CYBR_FIRE_V8_source.zip'
    archives=[archive_tree(staging,source)]
    # Test the actual extracted source bytes, using the declared generated cache
    # only as an integration fixture, not importing modules from the work tree.
    extracted_base=DATA/'v8-extracted-verification'
    if extracted_base.exists():shutil.rmtree(extracted_base)
    with zipfile.ZipFile(source) as archive:archive.extractall(extracted_base)
    extracted=extracted_base/'CYBR_FIRE_V8'
    if core_digest(extracted)!=capture['source_sha256']:raise AssertionError('Extracted core mismatch')
    (extracted/'cache').mkdir();(extracted/'cache/parent').symlink_to(ROOT/'cache/parent',target_is_directory=True)
    checks_dir=ROOT/'evidence-v8/extracted-source';checks_dir.mkdir(exist_ok=True)
    print('source extracted; running regressions',flush=True)
    commands=[('parent',[sys.executable,'-m','cybrfire','test']),
        ('native-fields',[sys.executable,'tests/test_native_fields.py']),
        ('banded-render',[sys.executable,'tests/test_streaming.py']),
        ('fine-transport',[sys.executable,'tests/test_detail.py']),
        ('fine-precision',[sys.executable,'tests/test_detail_precision.py']),
        ('fine-integration',[sys.executable,'tests/test_detail_integration.py'])]
    for name,command in commands:
        started=time.perf_counter();run(command,checks_dir/(name+'.log'),cwd=extracted)
        print('passed extracted',name,time.perf_counter()-started,flush=True)
    test_reports=[]
    names=['tests.json','extended-tests.json','runtime-tests.json','v5-tests.json',
        'v6-tests.json','particle-allocation-tests.json','native-field-tests.json','streaming-tests.json']
    for name in names:
        file=extracted/'evidence'/name;d=json.loads(file.read_text())
        if d.get('failed')!=0:raise AssertionError('Failed extracted test report '+name)
        shutil.copyfile(file,checks_dir/name);test_reports.append(dict(file=name,passed=d['passed'],failed=d['failed'],sha256=digest(file)))
    for name in ('detail-tests.json','detail-precision-tests.json','detail-integration-tests.json'):
        file=extracted/'evidence-v8'/name;d=json.loads(file.read_text())
        if d.get('failed')!=0:raise AssertionError('Failed extracted fine test report '+name)
        shutil.copyfile(file,checks_dir/name);test_reports.append(dict(file=name,passed=d['passed'],failed=d['failed'],sha256=digest(file)))
    parent=json.loads((ROOT/'cache/parent/manifest.json').read_text())
    speeds=[x['diagnostics']['max_component_speed'] for x in parent['frames'] if 'max_component_speed' in x['diagnostics']]
    budget=capture['fine_config']['trace_budget']*capture['fine_config']['trace_limit']
    observed=max(speeds)*(1/48)/state['spacing']
    rss=[]
    with (ROOT/'evidence-v8/capture-memory.csv').open() as stream:
        for row in csv.DictReader(stream):
            try:rss.append(int(row['process_high_water_kib']))
            except (ValueError,KeyError):pass
    tests_total=sum(x['passed'] for x in test_reports)
    validation=dict(release='CYBR FIRE V8 / fine scalar detail',status='completed_capture_and_extracted_source_tests_passed',
        verified_utc=datetime.now(timezone.utc).isoformat(),
        video=dict(file=video.name,sha256=digest(video),bytes=video.stat().st_size,probe=spec,full_decode=True),
        closeup=dict(file=detail_video.name,sha256=digest(detail_video),probe=detail_spec,full_decode=True,
            method='1:1 crop from the same native lossless frames',crop=[480,1720,1200,1600]),
        fine_scalar_grid=capture['fine_config']['shape'],fine_scalar_cells=99_090_432,scalar_storage='FP32',
        pressure_velocity_grid=capture['parent_grid'],pressure_velocity_cells=7_168_000,
        full_99_million_cell_CFD=False,modeled_subgrid_velocity=True,one_way_scalar_coupling=True,
        velocity_sampling_cache='FP16',fine_physical_spacing_m=state['spacing'],
        ember_particles_rendered=False,solid_obstacles_in_detail_layer=False,
        parent_cache_snapshot_advance_seconds=sum(x['seconds'] for x in parent['frames']),
        fine_time_excludes_parent_cache_build=True,
        native_video_dimensions=[2160,3840],fps=24,duration_seconds=1.,
        slow_motion=False,frame_interpolation=False,image_upscaling=False,render_time_density_warp=False,
        source_sha256=capture['source_sha256'],source_archive_sha256=digest(source),
        imported_initial_state_sha256=parent['input_sha256'],
        new_fine_steps=capture['final_fine_steps'],warmup_fine_steps=capture['warmup_frames']*capture['substeps'],
        first_frame_physical_time=frames[0]['time'],last_frame_physical_time=frames[-1]['time'],
        unique_lossless_frames=len(set(hashes)),mean_adjacent_frame_difference_8bit=float(np.mean(differences)),
        stills_are_original_video_frames_not_separate_renders=True,final_state=state,
        adapter=capture['adapter'],simulation_median_seconds_per_frame=float(np.median([f['simulation_seconds'] for f in frames])),
        rendering_median_seconds_per_frame=float(np.median([f['render_seconds'] for f in frames])),
        process_peak_RSS_KiB=max(rss) if rss else None,
        timing_qualification='Shared-container software OpenGL. No hardware GPU or matched full-engine benchmark.',
        extracted_source_tests=test_reports,extracted_source_passed=tests_total,extracted_source_failed=0,
        accuracy_warning=dict(maximum_sampled_parent_component_speed_m_s=max(speeds),
            parent_component_Courant_on_fine_step=observed,nominal_trace_work_budget=budget,
            exceeds_nominal_budget=observed>budget,
            qualification='Domain-wide parent sample, not measured characteristic error. Does not include modeled modes. Finer fields and positivity do not establish time/grid convergence.'),
        physical_calibration=False,EmberGen_comparison=False)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    # The source is unchanged after testing; reports live in the project and media.
    release=f'''# CYBR FIRE V8 — detail reconstruction\n\nCompleted native 2160 x 3840 capture, 24 new frames at 24 fps (one second of playback).\n\nThe fine stage evolves **99,090,432 FP32 scalar cells** (384 x 672 x 384, 5 mm spacing). The parent pressure/velocity solve has **7,168,000 cells**. Small-scale velocity is a modeled transverse spectrum, not resolved 99-million-cell CFD. Fine heat does not feed back into parent momentum. Velocity sampling caches use FP16.\n\nThe additional detail is transported through fuel, oxygen deficit, sensible energy and soot before rendering. There is no image generation, image sharpening, density warping at render time, upscaling, or slowed/frame-interpolated playback. The close-up is explicitly a 1:1 crop of the same lossless frames. The three PNG stills are original video frames, not independently rerendered shots.\n\n## Verified delivery\n\nBoth MP4s decode completely. All 24 source images are unique and match their recorded hashes. The source archive was extracted into a clean directory; **{tests_total} numerical/runtime checks passed**, including real cache/transport/render ownership checks using the declared generated parent cache as their integration fixture. These tests do not establish physical fidelity.\n\n## Measured limitations\n\nMedian work per output frame was {validation['simulation_median_seconds_per_frame']:.2f} seconds of fine simulation and {validation['rendering_median_seconds_per_frame']:.2f} seconds of rendering on Mesa software OpenGL. This is offline execution, not 24-fps real-time simulation. No hardware-GPU or EmberGen comparison is claimed.\n\nTransport is bounded, not exactly mass conservative. The manufactured Gaussian translation experiment measured about 0.52% mass drift in its cubic path. Large parent velocity peaks can exceed the nominal fine characteristic work budget; the report records this rather than treating finite images as a convergence result. The earlier parent-cache solve is additional work, reported separately in validation.json. The model is not experimentally calibrated. The detail layer currently excludes solid masks and ember coupling.\n\n## Reproduce\n\nThe project archive includes the original starting checkpoint under `input/v7-start.npz`, not the multi-gigabyte derived velocity cache. With Python, requirements.txt, Linux/EGL with OpenGL 4.5, and FFmpeg installed, run:\n\n```sh\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 LP_NUM_THREADS=4\npython tools/build_parent_cache.py input/v7-start.npz --out cache/parent\npython tools/capture_detail.py --parent-cache cache/parent --out captures/v8\n```\n\nSource-only users provide the earlier V7 complete ZIP or its start checkpoint. Different drivers are not promised bitwise-identical floating-point trajectories. The exact originally executed cache-builder script is preserved under baseline/.\n'''
    (OUT/'RELEASE.md').write_text(release)
    html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CYBR FIRE V8 | transported detail</title><style>body{background:#0d1119;color:#e5e9ef;font:16px/1.55 system-ui;margin:0 auto;padding:32px;max-width:1100px}h1{font-weight:600}p{max-width:850px}a{color:#a9d4ff}video{max-height:82vh;max-width:100%;background:#080c14}section{margin:40px 0}img{width:100%;height:auto}small{color:#adb7c4}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:20px}</style><h1>CYBR FIRE V8</h1><p>99.1 million transported scalar cells. 7.17 million parent flow cells. Native 4K, no upscaling. Small-scale motion is modeled, not full 99-million-cell CFD.</p><section><h2>Native detail close-up</h2><video controls loop playsinline src="V8_native_closeup.mp4" poster="V8_native_closeup.png"></video><p><small>A 1:1 crop of the original lossless frames. One second at 24 fps; no frame interpolation.</small></p></section><section><h2>Full native frame</h2><video controls loop playsinline src="V8_detail_rebuild_4K.mp4" poster="preview-12.jpg"></video></section><section><h2>Original-size stills</h2><div class="grid">'''
    for i in (0,12,23):html+=f'<a href="V8_detail_still_{i:02}.png"><img src="preview-{i:02}.jpg" alt="Original frame {i}"></a>'
    html+='</div></section><p><a href="RELEASE.md">Release and limitations</a> · <a href="validation.json">Executed validation</a> · <a href="capture.json">Frame-by-frame capture data</a></p>'
    (OUT/'index.html').write_text(html)
    media_stage=DATA/'v8-media-staging'/'CYBR_FIRE_V8_media';media_stage.mkdir(parents=True,exist_ok=True)
    for file in OUT.iterdir():
        if file.is_file() and (file.suffix in ('.mp4','.png','.jpg') or file.name in ('RELEASE.md','validation.json','capture.json','index.html','frame-journal.jsonl')) and not file.name.startswith('V8_encoding'):
            shutil.copyfile(file,media_stage/file.name)
    media=DATA/'CYBR_FIRE_V8_media.zip';archives.append(archive_tree(media_stage,media))
    # Reproducible project: executable source + actual starting state + evidence.
    project_stage=DATA/'v8-project-staging'/'CYBR_FIRE_V8'
    if project_stage.exists():shutil.rmtree(project_stage)
    shutil.copytree(staging,project_stage)
    (project_stage/'input').mkdir();shutil.copyfile(ROOT/'cache/parent/start.npz',project_stage/'input/v7-start.npz')
    shutil.copytree(media_stage,project_stage/'media')
    shutil.copytree(checks_dir,project_stage/'evidence-v8/extracted-source')
    for name in ('detail-tests-before-binding-repair.json','parent-cache.log','main-capture.log','capture-memory.csv'):
        shutil.copyfile(ROOT/'evidence-v8'/name,project_stage/'evidence-v8'/name)
    shutil.copyfile(ROOT/'cache/parent/manifest.json',project_stage/'evidence-v8/parent-manifest.json')
    (project_stage/'RELEASE.md').write_text(release)
    project=DATA/'CYBR_FIRE_V8_project.zip';archives.append(archive_tree(project_stage,project))
    packaging=dict(status='completed',source_extraction_tests_passed=tests_total,source_extraction_tests_failed=0,
        tested_archive_sha256=digest(source),tested_core_sha256=core_digest(extracted),archives=archives,
        source_only_requires_declared_start_cache=True,project_includes_start_checkpoint=True)
    (OUT/'packaging-validation.json').write_text(json.dumps(packaging,indent=2))
    (DATA/'CYBR_FIRE_V8_READY.json').write_text(json.dumps(packaging,indent=2))
    print('READY',json.dumps(packaging),flush=True)

if __name__=='__main__':main()
