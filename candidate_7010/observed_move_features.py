"""Observed offense of turn-30 active Pokemon, with official data only."""
import pandas as pd
from pokemon.features import type_mult

CATEGORIES=['active_p1','active_p2','active_type_matchup']

def make_frame(battles,dex,chart):
    cat_rows=[]
    for b in battles:
        st1=b['battle_timeline'][-1]['p1_pokemon_state'];st2=b['battle_timeline'][-1]['p2_pokemon_state']
        a=st1['name'];z=st2['name']
        if a in dex and z in dex:
            ab=max(type_mult(chart,t,dex[z]['types']) for t in dex[a]['types'] if t!='notype')
            ba=max(type_mult(chart,t,dex[a]['types']) for t in dex[z]['types'] if t!='notype')
            ty=f'{ab:g}|{ba:g}'
        else:
            ty='unknown'
        score={}
        for side,attacker,defender in [('p1',a,z),('p2',z,a)]:
            options={}
            if attacker in dex and defender in dex:
                for turn in b['battle_timeline']:
                    st=turn[f'{side}_pokemon_state']
                    move=turn[f'{side}_move_details']
                    if st['name']!=attacker or not move or move.get('base_power',0)<=0:
                        continue
                    off='atk' if move['category']=='physical' else 'spa'
                    deff='def' if move['category']=='physical' else 'spd'
                    power=float(move['base_power'])
                    multiplier=type_mult(chart,move['type'],dex[defender]['types'])
                    stab=1.5 if move['type'] in dex[attacker]['types'] else 1.0
                    ratio=min(2.5,max(0.5,float(dex[attacker][off])/max(1,dex[defender][deff])))
                    dmg=power*float(move.get('accuracy',1) or 1)*multiplier*stab*ratio
                    options[move['name']]=max(dmg,options.get(move['name'],0))
            score[side]=(float(max(options.values(),default=0)),len(options))
        cat_rows.append({'active_p1':a,'active_p2':z,'active_type_matchup':ty,
                         'seen_active_attack_p1':score['p1'][0],
                         'seen_active_attack_p2':score['p2'][0],
                         'n_seen_active_attacks_p1':score['p1'][1],
                         'n_seen_active_attacks_p2':score['p2'][1]})
    return pd.DataFrame(cat_rows)
