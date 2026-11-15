"""Summarize the traced kernels and CUDA API waits from an nsys SQLite export."""
import argparse
import json
from pathlib import Path
import sqlite3

parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
args=parser.parse_args()
con=sqlite3.connect(args.directory/'trace.sqlite')
names=dict(con.execute('SELECT id,value FROM StringIds'))
tables={row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
result={}
for kind,table,field in [('kernels','CUPTI_ACTIVITY_KIND_KERNEL','demangledName'),
                         ('runtime','CUPTI_ACTIVITY_KIND_RUNTIME','nameId'),
                         ('driver','CUPTI_ACTIVITY_KIND_DRIVER','nameId')]:
    if table not in tables:continue
    rows=list(con.execute(f'SELECT {field},COUNT(*),SUM(end-start),MIN(end-start),MAX(end-start) '
                          f'FROM {table} GROUP BY {field} ORDER BY SUM(end-start) DESC'))
    total=sum(r[2] for r in rows)
    result[kind]=[dict(name=names.get(name,str(name)),count=count,seconds=ns/1e9,
                       share_percent=100*ns/total if total else 0,
                       min_seconds=low/1e9,max_seconds=high/1e9)
                   for name,count,ns,low,high in rows]
(args.directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for kind,rows in result.items():
    print(kind)
    for row in rows[:15]:
        print(f"{row['seconds']:.6f}s {row['share_percent']:.2f}% {row['count']:6d} {row['name']}")
