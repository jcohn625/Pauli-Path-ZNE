"""Summarize absolute errors in O, U, ratio, and both normalized Q sectors."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
root=Path(__file__).resolve().parents[1];results=root/'results';summary=[]
for version in ('before','after'):
 data=json.loads((results/f'cnot_convergence_{version}.json').read_text())
 for engine in ('MPS','PEPS'):
  for pair in (0.,.0005):
   rows=[r for r in data['rows'] if r['engine']==engine and r['pair_rate']==pair]
   for cap in sorted({r['cap'] for r in rows}):
    group={(r['scale'],r['unsigned']):r for r in rows if r['cap']==cap}
    def quantities(field):
     o0=group[(0.,False)][field];u0=group[(0.,True)][field]
     o0=o0 if abs(o0)>=1e-12 else float('nan')
     plus=u0+group[(0.,False)][field];minus=u0-group[(0.,False)][field]
     plus=plus if abs(plus)>=1e-12 else float('nan');minus=minus if abs(minus)>=1e-12 else float('nan')
     return {name:np.array(values) for name,values in {
       'O':[group[(s,False)][field] for s in (0.,1.,3.)],
       'U':[group[(s,True)][field] for s in (0.,1.,3.)],
       'ratio':[group[(s,False)][field]/o0 for s in (0.,1.,3.)],
       'Q+':[(group[(s,True)][field]+group[(s,False)][field])/plus for s in (0.,1.,3.)],
       'Q-':[(group[(s,True)][field]-group[(s,False)][field])/minus for s in (0.,1.,3.)]}.items()}
    exact=quantities('exact')
    errors={field:{key:(float(np.max(abs(value-exact[key]))) if np.isfinite(value).all() and np.isfinite(exact[key]).all() else None) for key,value in quantities(field).items()}
       for field in ('approximate','corrected')}
    row=dict(version=version,engine=engine,pair_rate=pair,cap=cap,
      curve_seconds=sum(r['median_s'] for r in group.values()),errors=errors)
    summary.append(row)
(results/'cnot_convergence_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for engine in ('MPS','PEPS'):
 for pair in (0.,.0005):
  for tol in (1e-3,1e-5,1e-8):
   print(engine,pair,'tolerance',tol)
   for version in ('before','after'):
    passing=[r for r in summary if r['engine']==engine and r['pair_rate']==pair and r['version']==version
      and max(float('inf') if x is None else x for x in r['errors']['corrected'].values())<=tol]
    best=min(passing,key=lambda r:r['cap']) if passing else None
    print(version, None if best is None else (best['cap'],round(best['curve_seconds'],4),best['errors']['corrected']))
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for i,engine in enumerate(('MPS','PEPS')):
 for j,pair in enumerate((0.,.0005)):
  ax=axes[i,j]
  for version,color in [('before','#bd532c'),('after','#1769aa')]:
   group=[r for r in summary if r['engine']==engine and r['pair_rate']==pair and r['version']==version]
   for field,style in [('approximate','--'),('corrected','-')]:
    ax.plot([r['cap'] for r in group],[(float('nan') if any(x is None for x in r['errors'][field].values()) else max(1e-16,max(r['errors'][field].values()))) for r in group],
      style,marker='o',color=color,label=f'{version}, {field}')
  ax.set(xscale='log',yscale='log',xlabel='Spatial / MPS bond cap',ylabel='Maximum absolute error',
         title=f"{engine}: {'single + pair noise' if pair else 'single noise'}")
  ax.set_xticks([2,4,8,16,32],labels=['2','4','8','16','32']);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Six-qubit exact-reference convergence\nMaximum over O, U, ratio, Q± and noise scales 0, 1, 3')
fig.savefig(results/'cnot_convergence.png',dpi=170)
