"""Effective speed and observed priority at turn 30; official input only."""
import numpy as np
import pandas as pd

COLUMNS=['speed_effective_difference','speed_effective_ratio','speed_p1_first',
         'priority_observed_difference','speed_threat_difference']

def row(battle,dex):
    timeline=battle['battle_timeline']
    s1=timeline[-1]['p1_pokemon_state']
    s2=timeline[-1]['p2_pokemon_state']
    def speed_priority(side,state):
        name=state['name']
        base=dex.get(name,{}).get('spe',75)
        stage=state['boosts'].get('spe',0)
        stage_factor=(2+stage)/2 if stage>=0 else 2/(2-stage)
        effective=base*stage_factor*(0.25 if state['status']=='par' else 1.0)
        priorities=[
            move.get('priority',0) for turn in timeline
            if turn[f'{side}_pokemon_state']['name']==name
            for move in [turn[f'{side}_move_details']] if move is not None
        ]
        return effective,max(priorities,default=0)
    speed1,priority1=speed_priority('p1',s1)
    speed2,priority2=speed_priority('p2',s2)
    if priority1>priority2 or (priority1==priority2 and speed1>speed2):
        moves_first=1.0
    elif priority1==priority2 and speed1==speed2:
        moves_first=0.5
    else:
        moves_first=0.0
    return {
        'speed_effective_difference':speed1-speed2,
        'speed_effective_ratio':np.log((speed1+1)/(speed2+1)),
        'speed_p1_first':moves_first,
        'priority_observed_difference':priority1-priority2,
        'speed_threat_difference':float(speed1>speed2),
    }

def make_frame(battles,dex):
    return pd.DataFrame([row(b,dex) for b in battles],columns=COLUMNS)
