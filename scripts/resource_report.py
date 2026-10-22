"""Measured resource account for a completed local experiment."""
import argparse,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from SaTP.common import dump,digest

def events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    parent=(ROOT/args.run).resolve()
    if not parent.is_relative_to(ROOT):raise ValueError('Outside project')
    status=json.loads((parent/'status.json').read_text())
    controls=json.loads((parent/'ablations_seed42/manifest.json').read_text())
    if status['state']!='completed' or controls['state']!='completed':raise ValueError('Incomplete execution')
    main_events=events(parent/'events.jsonl');ab_events=events(parent/'ablations_seed42/events.jsonl')
    validation=[{key:e[key] for key in ['time','solution','seed','params']} for e in main_events if e['event']=='validation']
    test=[dict(seed=e['seed'],solution=e['solution'],native_evaluation_seconds=e['metrics']['evaluation_seconds']) for e in main_events if e['event']=='test_complete']
    # Storage is observed final disk use, not a disk peak. Host RAM peak was not instrumented.
    storage={}
    for name in ['.python','.venv','.cache','data','experiments','artifacts','reports']:
        storage[name]=sum(p.stat().st_size for p in (ROOT/name).rglob('*') if p.is_file()) if (ROOT/name).exists() else 0
    rm_seconds=status['seconds']
    environment_script=ROOT/'project_env.ps1'
    start=environment_script.stat().st_ctime if environment_script.exists() else None
    supplementary_seconds=None
    if ab_events:
        from datetime import datetime
        supplementary_seconds=(datetime.fromisoformat(ab_events[-1]['time'])-datetime.fromisoformat(ab_events[0]['time'])).total_seconds()
    previous=ROOT/'experiments/2402b7dfbe7c_a9e276bb/status.json'
    old=json.loads(previous.read_text()) if previous.exists() else {}
    superseded_span=previous.stat().st_mtime-old['start_unix'] if old.get('start_unix') is not None else None
    baseline_manifest=parent/'baseline_parent.json'
    baseline_parent=json.loads(baseline_manifest.read_text()) if baseline_manifest.exists() else None
    result=dict(primary_matrix_wall_seconds=rm_seconds,primary_matrix_hours=rm_seconds/3600,
        peak_primary_cuda_allocated_bytes=status['peak_cuda_bytes'],host_ram_peak_bytes=None,
        host_ram_peak_note='not instrumented; do not interpret as zero',
        scope='Main matrix includes RM fitting, validation search and native/common evaluation. Reused baseline and superseded-run costs, when present, are listed separately.',
        superseded_before_test_elapsed_seconds=superseded_span,superseded_timing_note='Elapsed from recorded runner start to superseded status write; includes engineering interruptions, not GPU-active seconds.',
        baseline_parent=baseline_parent,
        elapsed_since_environment_script_created_hours=(time.time()-start)/3600 if start is not None else None,
        supplementary_first_to_last_logged_event_seconds=supplementary_seconds,
        supplementary_timing_note='Event span, not total end-to-end time; script startup, final statistical tables and notebook execution excluded.',
        validation_trials=validation,native_test_timings=test,final_storage_bytes=storage,storage_peak_bytes=None,
        completed_supplementary_cells=controls['cells'],source_script_sha256=digest(__file__))
    dump(parent/'resources.json',result)
    print(json.dumps({key:result[key] for key in ['primary_matrix_hours','peak_primary_cuda_allocated_bytes','elapsed_since_environment_script_created_hours','completed_supplementary_cells']}))
if __name__=='__main__':main()
