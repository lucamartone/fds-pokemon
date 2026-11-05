"""Symmetric battle-state features based exclusively on official observations.

The two side representations can be exchanged without accessing hidden
opponent species, future turns, test labels or external data.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from pokemon.features.base import (
    damage_features,
    load_type_chart,
    read_jsonl,
    side_features,
    type_mult,
)
from pokemon.features.resources import build_dex

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
HEAL = {'rest', 'recover', 'softboiled', 'milkdrink', 'moonlight', 'morningsun', 'synthesis'}
SETUP = {'curse', 'swordsdance', 'agility', 'growth', 'amnesia', 'bellydrum', 'barrier', 'acidarmor'}
STATUS_MOVES = {'toxic', 'thunderwave', 'sleeppowder', 'stunspore', 'spore', 'hypnosis', 'lovelykiss', 'glare', 'poisonpowder', 'sing'}
STATUS_COST = {'nostatus':1., 'par':.75, 'brn':.8, 'psn':.72, 'tox':.67, 'slp':.65, 'frz':.40, 'fnt':0.}
CATEGORIES = ['active_a', 'active_b', 'status_a', 'status_b', 'lead_a', 'lead_b']


def observed_side(battle: dict, side: str, dex: dict, default: dict) -> dict:
    """Summarize a side without assuming anything about hidden P2 identities."""
    timeline=battle['battle_timeline']
    sk=f'{side}_pokemon_state'; mk=f'{side}_move_details'
    original_side, _ = side_features(timeline, side)
    observed_names=list(dict.fromkeys(t[sk]['name'] for t in timeline))
    full_names=[p['name'] for p in battle['p1_team_details']] if side=='p1' else observed_names
    last={t[sk]['name']: t[sk] for t in timeline}
    hidden_count=max(0,6-len(full_names))
    known_names=[n for n in full_names if n in dex]
    alive_names=[n for n in known_names if n not in last or last[n]['hp_pct']>0]
    hp={n:float(last[n]['hp_pct']) if n in last else 1. for n in known_names}
    status={n:last[n]['status'] if n in last else 'nostatus' for n in known_names}
    current=timeline[-1][sk]
    moves=[t[mk] for t in timeline if t[mk] is not None]
    movement=Counter(m['name'] for m in moves)
    active=current['name']
    active_moves={m['name']:m for t in timeline for m in [t[mk]] if m and t[sk]['name']==active}
    dmg_dealt,hits_dealt=damage_features(timeline,'p2' if side=='p1' else 'p1')
    dmg_received,hits_received=damage_features(timeline,side)
    hp_seq=np.array([t[sk]['hp_pct'] for t in timeline],float)
    names=[t[sk]['name'] for t in timeline]
    changes=[i for i in range(1,30) if names[i]!=names[i-1]]
    same_mon=np.array([names[i]==names[i-1] for i in range(1,30)])
    d=np.diff(hp_seq)[same_mon]
    benefits=np.clip(d,0,None)
    losses=np.clip(-d,0,None)
    f={k:float(v) for k,v in original_side.items()}
    f.update({
      'roster_known':float(len(full_names)),
      'roster_observed':float(len(observed_names)),
      'roster_missing':float(hidden_count),
      'known_alive':float(len(alive_names)),
      'hp_known_sum':sum(hp.values()),
      'hp_known_low':sum(0<hp[n]<.35 for n in known_names),
      'hp_known_healthy':sum(hp[n]>.7 for n in known_names),
      'hp_known_min_alive':min([hp[n] for n in alive_names],default=0.),
      'hp_known_min_all':min(hp.values(),default=0.),
      'hp_known_std':float(np.std(list(hp.values()))) if hp else 0.,
      'hp_effective_known':sum(hp[n]*STATUS_COST.get(status[n],1.) for n in known_names),
      'hp_effective_all':sum(hp[n]*STATUS_COST.get(status[n],1.) for n in known_names)+hidden_count,
      'status_known':sum(status[n] not in ('nostatus','fnt') for n in known_names),
      'damage_dealt':dmg_dealt,
      'damage_received':dmg_received,
      'hits_dealt':hits_dealt,
      'hits_received':hits_received,
      'recover_gain':float(np.sum(benefits)),
      'damage_loss':float(np.sum(losses)),
      'switch_count_first10':sum(i<=9 for i in changes),
      'switch_count_last10':sum(i>=20 for i in changes),
      'last_switch_age':29-max(changes,default=0),
      'last_hp_slope':float(np.polyfit(np.arange(10),hp_seq[-10:],1)[0]),
      'heal_move_total':sum(m['name'] in HEAL for m in moves),
      'setup_move_total':sum(m['name'] in SETUP for m in moves),
      'status_move_total':sum(m['name'] in STATUS_MOVES for m in moves),
      'unique_moves':len(movement),
      'unique_active_moves':len(active_moves),
      'mean_attack_power':float(np.mean([m['base_power'] for m in moves])) if moves else 0.,
      'max_attack_power':max([m['base_power'] for m in moves],default=0.),
      'active_attack_power':max([m['base_power'] for m in active_moves.values()],default=0.),
      'active_speed':float(dex.get(active,default)['spe']),
      'active_bst':float(sum(dex.get(active,default)[s] for s in STATS)),
      'active_effective_hp':float(current['hp_pct'])*STATUS_COST.get(current['status'],1.),
      'active_turns':sum(n==active for n in names),
      'active_setup_known':sum(m['name'] in SETUP for m in active_moves.values()),
      'active_heal_known':sum(m['name'] in HEAL for m in active_moves.values()),
      'active_status_known':sum(m['name'] in STATUS_MOVES for m in active_moves.values()),
    })
    for st in STATS:
        values=[dex[n][st] for n in alive_names]
        f[f'stats_{st}_alive_sum']=sum(values)+hidden_count*default[st]
        f[f'stats_{st}_hp_weighted']=sum(hp[n]*dex[n][st] for n in alive_names)+hidden_count*default[st]
        f[f'stats_{st}_mean']=float(np.mean([dex[n][st] for n in known_names])) if known_names else default[st]
        f[f'stats_{st}_max']=max([dex[n][st] for n in known_names],default=default[st])
    return {'values':f,'alive':alive_names,'last':last,'hp':hp,'active':active,'active_state':current,'moves':active_moves,
            'lead':battle['battle_timeline'][0][sk]['name'], 'known':known_names}


def attack_quality(side:dict, target:dict, dex:dict, chart:dict)->dict:
    """Calculate observed attack pressure and type coverage against a defender."""
    attacker=side['active'];defender=target['active']
    if attacker not in dex or defender not in dex:
        return {'best_attack':0.,'attack_coverage':0.,'n_effective_moves':0.}
    powers=[]
    for mv in side['moves'].values():
        if mv['base_power']<=0:continue
        offense='atk' if mv['category']=='physical' else 'spa'
        defense='def' if mv['category']=='physical' else 'spd'
        ratio=float(dex[attacker][offense])/max(float(dex[defender][defense]),1.)
        ratio=min(2.5,max(.5,ratio))
        stab=1.5 if mv['type'] in dex[attacker]['types'] else 1.
        multiplier=type_mult(chart,mv['type'],dex[defender]['types'])
        powers.append(mv['base_power']*mv['accuracy']*stab*multiplier*ratio)
    def statab(a,b):
        if a not in dex or b not in dex:return 1.
        return max(type_mult(chart,t,dex[b]['types']) for t in dex[a]['types'] if t!='notype')
    if side['alive'] and target['alive']:
        mx=np.array([[statab(a,b) for b in target['alive']] for a in side['alive']],float)
        coverage=np.mean(mx.max(0)>=2.)
    else:coverage=0.
    return {'best_attack':float(max(powers,default=0.)),
            'attack_coverage':float(coverage),
            'n_effective_moves':int(sum(v>=70 for v in powers))}


def encode_battle(b: dict,dex:dict,chart:dict,default:dict)->dict:
    """Return two interchangeable side descriptions and pairwise interaction."""
    a=observed_side(b,'p1',dex,default);c=observed_side(b,'p2',dex,default)
    am=attack_quality(a,c,dex,chart);cm=attack_quality(c,a,dex,chart)
    for key,value in am.items():a['values'][key]=value
    for key,value in cm.items():c['values'][key]=value
    for side,op in ((a,c),(c,a)):
       current=side['active_state'];opcurrent=op['active_state']
       side['values']['speed_advantage']=float(side['values']['active_speed']>op['values']['active_speed'])
       side['values']['active_hp_advantage']=float(current['hp_pct']-opcurrent['hp_pct'])
       side['values']['effective_attack_vs_enemy_hp']=side['values']['best_attack']/(0.15+float(opcurrent['hp_pct']))
    return a,c


def orientation(a:dict,b:dict,te_a=None,te_b=None)->dict:
    """Encode a battle orientation with explicit anti-symmetric differences."""
    d={}
    for key,v in a['values'].items():
        w=b['values'][key]
        d[f'd_{key}']=v-w
        d[f's_{key}']=v+w
    d['active_a']=a['active'];d['active_b']=b['active']
    d['status_a']=a['active_state']['status'];d['status_b']=b['active_state']['status']
    d['lead_a']=a['lead'];d['lead_b']=b['lead']
    if te_a is not None:
        d['d_team_prior']=float(te_a[0]-te_b[0])
        d['s_team_prior']=float(te_a[0]+te_b[0])
        d['d_team_prior_games']=float(te_a[1]-te_b[1])
        d['s_team_prior_games']=float(te_a[1]+te_b[1])
    return d


def prepare(data_dir:Path):
    """Extract feature dictionaries and target-independent species information."""
    t0=monotonic()
    tr=read_jsonl(data_dir/'train.jsonl')
    test=read_jsonl(data_dir/'test.jsonl')
    chart=load_type_chart(data_dir/'type_chart_gen2.csv')
    dex=build_dex(tr,test)
    default={k:float(np.mean([x[k] for x in dex.values()])) for k in STATS}
    pairs=[encode_battle(b,dex,chart,default) for b in tr+test]
    labels=np.asarray([b['player_won'] for b in tr],int)
    if len(tr)!=10000 or len(test)!=5000:raise ValueError('Unexpected data shape')
    print('Computed symmetrical features for',len(pairs),'battles in',round(monotonic()-t0,2),'s',flush=True)
    return pairs,labels,[b['battle_id'] for b in test]


def get_prior(te:pd.DataFrame,index:int)->tuple[float,float]:
    """Obtain folded original-team and inferred opponent-team win-rate priors."""
    r=te.iloc[index]
    return (float(r['te_p1_wr']),float(r['te_p1_n'])), (float(r['te_p2c_wr']),float(r['te_p2c_n']))


def create_pair_matrix(pairs,indices,te,swap=False):
    """Construct one labeled orientation per battle for training/inference."""
    rows=[]
    for pos,j in enumerate(indices):
        a,b=pairs[j]
        ta,tb=get_prior(te,pos)
        rows.append(orientation(b,a,tb,ta) if swap else orientation(a,b,ta,tb))
    return pd.DataFrame(rows)


def model(seed=42):
    """Fixed CatBoost model; hyperparameters are not selected via Kaggle."""
    return CatBoostClassifier(iterations=550,depth=6,learning_rate=.035,
        l2_leaf_reg=12,rsm=.8,random_seed=seed,thread_count=4,
        cat_features=CATEGORIES,verbose=False,allow_writing_files=False)

