"""Observed recovery, disruption and setup moves by turn-30 active Pokemon."""
import pandas as pd
RECOVERY={'rest','recover','softboiled','milkdrink','moonlight','morningsun','synthesis'}
STATUS={'toxic','thunderwave','stunspore','spore','sleeppowder','hypnosis','lovelykiss','sing','glare','poisonpowder','willowisp'}
SETUP={'curse','swordsdance','growth','agility','amnesia','bellydrum','dragondance','calmmind','irondefense','nastyplot','bulkup','barrier','acidarmor'}
COLUMNS=[f'{side}_active_{kind}' for side in ('p1','p2') for kind in ('recovery','disrupt','setup')]

def row(battle):
    timeline=battle['battle_timeline']
    out={}
    for side in ('p1','p2'):
        name=timeline[-1][f'{side}_pokemon_state']['name']
        moves={mv['name'] for t in timeline
               if t[f'{side}_pokemon_state']['name']==name
               for mv in [t[f'{side}_move_details']] if mv is not None}
        out.update({f'{side}_active_recovery':float(bool(moves & RECOVERY)),
                    f'{side}_active_disrupt':float(bool(moves & STATUS)),
                    f'{side}_active_setup':float(bool(moves & SETUP))})
    return out

def make_frame(battles):
    return pd.DataFrame([row(b) for b in battles],columns=COLUMNS)
