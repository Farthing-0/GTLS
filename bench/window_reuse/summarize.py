"""Summarize the saved full-flow timings and nsys SQLite kernel measurements."""
import json
import csv
import importlib.util
import math
from pathlib import Path
import sqlite3

ROOT=Path('/home/farthing/GTLS/GTLS-Compare/tls2-gtls-delta001-20261114/round1-window-reuse-20261114')


def trace(path):
    db=sqlite3.connect(f'file:{path}/trace.sqlite?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    kernels=[dict(r) for r in db.execute('''
        SELECT s.value name, COUNT(*) launches, SUM(k.end-k.start)/1e9 seconds,
        MIN(k.registersPerThread) registers_min, MAX(k.registersPerThread) registers_max,
        MAX(k.localMemoryPerThread) local_bytes, MIN(k.gridY) min_periods, MAX(k.gridY) max_periods
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
        GROUP BY s.value ORDER BY seconds DESC''')]
    copies=[dict(r) for r in db.execute('''SELECT copyKind, SUM(bytes) bytes,
        SUM(end-start)/1e9 seconds FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind''')]
    full=db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text LIKE 'GTLS_round1_%' AND end IS NOT NULL").fetchone()
    intervals=[]
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        intervals.extend((max(s,full['start']),min(e,full['end'])) for s,e in db.execute(
            f'SELECT start,end FROM {table}') if e>full['start'] and s<full['end'])
    merged=[]
    for s,e in sorted(intervals):
        if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
        else: merged.append([s,e])
    seconds=(full['end']-full['start'])/1e9
    return dict(kernels=kernels,copies=copies,nvtx_seconds=seconds,
                kernel_seconds=sum(k['seconds'] for k in kernels),
                search_seconds=sum(k['seconds'] for k in kernels if k['name'].startswith('searchChi2')),
                gpu_busy_percent=sum(e-s for s,e in merged)/1e9/seconds*100)


def main():
    summary={}
    for path in sorted(ROOT.glob('*/result.json')):
        data=json.loads(path.read_text())
        if 'median_seconds' in data:
            record={k:data[k] for k in ('input','median_seconds','speedup','all_fields_equal')}
            record['telemetry']={}
            for variant in ('baseline','tail','optimized'):
                samples=[s for r in data['runs'] if r['variant']==variant for s in r.get('telemetry',[])]
                if samples:
                    busy=[s for s in samples if s['gpu_percent']>80] or samples
                    record['telemetry'][variant]=dict(max_temperature=max(s['temperature_c'] for s in samples),
                        busy_sm_mhz_min=min(s['sm_mhz'] for s in busy),busy_sm_mhz_max=max(s['sm_mhz'] for s in busy),
                        throttle_bits=sorted(set(s['throttle_bits'] for s in samples)))
            summary[path.parent.name]=record
    for variant in ('baseline','optimized'):
        path=ROOT/f'nsys-{variant}'
        if (path/'trace.sqlite').exists(): summary['nsys-'+variant]=trace(path)
    parser_path=ROOT.parent/'bottleneck-analysis-20261114/analyze_ncu.py'
    spec=importlib.util.spec_from_file_location('_round1_ncu',parser_path)
    parser=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    counter_rows=[]
    names={
        'l1_wavefront_percent_peak':'l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed',
        'l1_requests_percent_peak':'l1tex__lsuin_requests.avg.pct_of_peak_sustained_elapsed',
        'l2_percent_peak':'lts__throughput.avg.pct_of_peak_sustained_elapsed',
        'dram_percent_peak':'dram__cycles_active.avg.pct_of_peak_sustained_elapsed',
        'fma_percent_peak':'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed',
        'occupancy_percent':'sm__warps_active.avg.pct_of_peak_sustained_active',
        'eligible_warps_per_scheduler':'smsp__warps_eligible.avg.per_cycle_active',
        'issue_active_percent':'smsp__issue_active.avg.pct_of_peak_sustained_active',
    }
    for variant in ('baseline','optimized'):
        path=ROOT/f'ncu-{variant}'/'metrics.csv'
        if not path.exists(): continue
        valid=[]
        for kernel in parser.read_metrics(path).values():
            m=kernel['metrics']
            duration=m.get('gpu__time_duration.sum',{}).get('value')
            if not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
                continue
            record=dict(variant=variant,kernel=kernel['name'])
            for label,metric in names.items(): record[label]=m.get(metric,{}).get('value')
            denominator=m['smsp__average_warp_latency_per_inst_issued.ratio']['value']
            for label in ('lg_throttle','long_scoreboard'):
                metric='smsp__average_warps_issue_stalled_'+label+'_per_issue_active.ratio'
                record[label+'_percent_warp_interval']=100*m[metric]['value']/denominator
            valid.append(record)
        expected={f'searchChi2{"Adjacent" if variant=="optimized" else ""}G{g}' for g in (8,16,32)}
        assert {k['kernel'] for k in valid}==expected, 'Incomplete ncu capture: '+variant
        counter_rows.extend(valid)
        summary['ncu-'+variant]=valid
    if counter_rows:
        with (ROOT/'hardware_counter_comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(counter_rows[0]))
            writer.writeheader();writer.writerows(counter_rows)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,data in summary.items():
        print(name,json.dumps({k:v for k,v in data.items() if k not in ('kernels','all_fields_equal','input')}
                             if isinstance(data,dict) else data))


if __name__=='__main__': main()
