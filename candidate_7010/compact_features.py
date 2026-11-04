"""Compact turn-30 resource features; official competition inputs only."""
import numpy as np
from pokemon.features import type_mult

STATUS_COST={'nostatus':1.0,'par':0.78,'brn':0.8,'psn':0.78,'tox':0.72,'slp':0.65,'frz':0.5,'fnt':0.0}
HEAL={'rest','recover','softboiled','milkdrink','moonlight','morningsun','synthesis'}
STATS=['hp','atk','def','spa','spd','spe']

def side(b, name, dex):
    t=b['battle_timeline'];state=f'{name}_pokemon_state';mv=f'{name}_move_details'
    hist={};times={};observed=[]
    for j,turn in enumerate(t):
        st=turn[state];pn=st['name'];hist[pn]=st;times[pn]=j
        if pn not in observed:observed.append(pn)
    all_names=[p['name'] for p in b['p1_team_details']] if name=='p1' else observed
    unshown=0 if name=='p1' else max(0,6-len(all_names))
    hps=np.array([float(hist[n]['hp_pct']) if n in hist else 1.0 for n in all_names]+[1.0]*unshown)
    factors=np.array([STATUS_COST.get(hist[n]['status'],1.) if n in hist else 1. for n in all_names]+[1.0]*unshown)
    ehp=hps*factors
    alive=np.array([h for h in hps if h>0])
    team_known=[n for n in all_names if n in dex]
    valid_alive=[n for n in team_known if (n not in hist or hist[n]['hp_pct']>0)]
    last_moves=[t0[mv]['name'] for t0 in t if t0[mv]]
    active=t[-1][state]
    l10=[float(t0[state]['hp_pct']) for t0 in t[-10:]]
    unique_ko=sum(hist[n]['status']=='fnt' or hist[n]['hp_pct']==0 for n in observed)
    score={
        'hp':float(np.sum(hps)),'effective_hp':float(np.sum(ehp)),'alive':int(np.sum(hps>0)),
        'fainted':unique_ko,'critical':int(np.sum((hps>0)&(hps<.3))),
        'below_half':int(np.sum((hps>0)&(hps<.5))),
        'healthy':int(np.sum(hps>=.8)),
        'hp_low3':float(np.sort(hps)[:3].sum()),'hp_high3':float(np.sort(hps)[-3:].sum()),
        'hp_std':float(np.std(hps)),'hp_min_alive':float(np.min(alive)) if len(alive) else 0.,
        'status_weight_loss':float(np.sum(hps-ehp)),
        'statused':int(np.sum((hps>0)&(factors<.95))),
        'active_hp':float(active['hp_pct']),
        'active_effective_hp':float(active['hp_pct'])*STATUS_COST.get(active['status'],1.),
        'active_boost':float(sum(active['boosts'].values())),
        'active_speed_boost':float(active['boosts']['spe']),
        'last10_active_hp_mean':float(np.mean(l10)),
        'heal_used':sum(k in HEAL for k in last_moves),
        'heal_seen':int(any(k in HEAL for k in last_moves)),
        'last5_heal_used':sum((t0[mv] or {}).get('name') in HEAL for t0 in t[-5:]),
        'unshown':unshown,'shown':len(observed),
        'last_switch_age':29-next((j for j in range(29,0,-1) if t[j][state]['name']!=t[j-1][state]['name']),0),
    }
    for stat in ('atk','spa','def','spd','spe'):
        score['surviving_'+stat]=float(sum(dex[n][stat]*(hist[n]['hp_pct'] if n in hist else 1.)*STATUS_COST.get(hist[n]['status'],1.) if n in hist else dex[n][stat] for n in valid_alive))
    return score,valid_alive,hist

def row(b,dex,chart):
    s1,a,h1=side(b,'p1',dex);s2,z,h2=side(b,'p2',dex)
    out={'p2_unshown':s2['unshown'],'p2_shown':s2['shown']}
    for key in s1:
        if key not in ('unshown','shown'):
            out['diff_'+key]=s1[key]-s2[key]
    for key in ['hp','effective_hp','status_weight_loss','alive','fainted','heal_seen','active_hp','active_boost']:
        out['p1_'+key]=s1[key];out['p2_'+key]=s2[key]
    if a and z:
        def beststab(p,q):
            return max(type_mult(chart,at,dex[q]['types']) for at in dex[p]['types'] if at!='notype')
        m1=np.array([[beststab(p,q) for q in z] for p in a]);m2=np.array([[beststab(q,p) for q in z] for p in a])
        w1=np.array([(h1[n]['hp_pct'] if n in h1 else 1.)*STATUS_COST.get(h1[n]['status'],1.) if n in h1 else 1. for n in a])
        w2=np.array([h2[n]['hp_pct']*STATUS_COST.get(h2[n]['status'],1.) for n in z])
        off=(m1.max(axis=0)*w2).sum();defensive=(m2.max(axis=1)*w1).sum()
        out['matchup_offense_minus_defense']=float(off-defensive)
        out['matchup_offense']=float(off);out['matchup_defense']=float(defensive)
        out['coverage_p1']=float(np.mean(m1.max(axis=0)>=2))
        out['coverage_p2']=float(np.mean(m2.max(axis=1)>=2))
    else:
        out.update({k:0. for k in ('matchup_offense_minus_defense','matchup_offense','matchup_defense','coverage_p1','coverage_p2')})
    return out
