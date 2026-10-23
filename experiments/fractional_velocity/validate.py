"""Calibrate the whole measurement pipeline on perfect predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run import digit_data, make_case, measure, SPEEDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=32)
    p.add_argument('--out',type=Path,default=Path('outputs/fractional_velocity/calibration'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    images,labels=digit_data();rows=[]
    for seed in range(args.seeds):
        for mode in ['nearest','bilinear','half_offset']:
            for v in SPEEDS if mode!='half_offset' else [0,1,2]:
                case=make_case(seed,v,mode,images,labels)
                measurements,_=measure(case,case['frames'][50:])
                rows.extend(measurements)
        print(f'calibration {seed+1}/{args.seeds}',flush=True)
    x=pd.DataFrame(rows);x.to_csv(args.out/'perfect_prediction_metrics.csv',index=False)
    x=x[x.horizon==40].copy()
    summary={}
    for mode,g in x.groupby('mode'):
        summary[mode]=dict(velocity_mae=float(abs(g.predicted_vx-g.speed).mean()),
            tracker_velocity_mae=float(abs(g.tracker_vx-g.speed).mean()),
            trajectory_error=float(g.ade.mean()),tracker_trajectory_error=float(g.tracker_ade.mean()),
            missing_rate=float(g.missing_frames.sum()/g.visible_frames.sum()),
            ghost_rate=float(g.ghost_frames.sum()/g.visible_frames.sum()),
            coverage=float(g.scored_frames.sum()/g.visible_frames.sum()),
            velocity_fit_rate=float(g.predicted_vx.notna().mean()))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)


if __name__=='__main__':main()
