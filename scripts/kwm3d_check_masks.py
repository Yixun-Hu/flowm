#!/usr/bin/env python3
"""Validate 20 predetermined frames against saturated block colours; CPU only."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / 'algorithms/mem_wm/backbones/kwm3d/.scratch'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0,str(ROOT))


def check_masks(data_root=None, output=None):
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from PIL import Image, ImageDraw
    from algorithms.mem_wm.backbones.kwm3d.metrics import project_block_masks, saturation_mask
    data_root=Path(data_root or ROOT/'data/blockworld/k3d_v_validation')
    paths=sorted(data_root.glob('*/*_rgb.mp4'))[:20]
    if len(paths)<4:
        raise FileNotFoundError('Need four k3d_v_validation clips for the 20-frame control')
    tiles,records=[],[]
    empty_controls = 0
    # Prespecified temporal samples, not selected for colour overlap or IoU.
    times=[10,30,50,70,90]
    for path in paths:
        info=torch.load(path.with_name(path.stem.removesuffix('_rgb')+'_block_info.pt'),map_location='cpu',weights_only=False)
        frames=torch.from_numpy(VideoReader(str(path),ctx=cpu(),num_threads=1).get_batch(times).asnumpy()).permute(0,3,1,2).float()/255
        coloured=torch.tensor([str(c).lower() not in ('grey','gray','none') for c in info['block_colors']])
        masks=project_block_masks(info['block_pos_agent'][times],coloured.expand(len(times),-1))
        for t,frame,mask in zip(times,frames,masks):
            control=saturation_mask(frame)
            # A colour control cannot validate an empty image. Select only by
            # independently observed colour presence, never by mask agreement.
            if not control.any():
                empty_controls += 1
                continue
            intersection=int((mask&control).sum()); union=int((mask|control).sum())
            iou=intersection/union if union else None
            records.append(dict(clip=str(path.relative_to(data_root)),frame=t,iou=iou,
                                analytic_pixels=int(mask.sum()),saturated_pixels=int(control.sum())))
            rgb=(frame.permute(1,2,0).numpy()*255).astype(np.uint8)
            overlay=rgb.astype(float)
            overlay[mask.numpy()]=.55*overlay[mask.numpy()]+.45*np.array([0,255,255])
            overlay[control.numpy()]=.7*overlay[control.numpy()]+.3*np.array([255,0,255])
            tile=Image.new('RGB',(256,150),'white')
            tile.paste(Image.fromarray(rgb),(0,22));tile.paste(Image.fromarray(overlay.astype(np.uint8)),(128,22))
            ImageDraw.Draw(tile).text((3,4),f'{path.parent.name}/{path.stem} t={t} IoU={iou:.3f}' if iou is not None else f't={t} empty',fill='black')
            tiles.append(tile)
            if len(records) == 20:
                break
        if len(records) == 20:
            break
    if len(records) != 20:
        raise ValueError('Fewer than 20 frames with visible saturated colours')
    montage=Image.new('RGB',(256*4,150*5),'white')
    for i,tile in enumerate(tiles): montage.paste(tile,((i%4)*256,(i//4)*150))
    output=Path(output or SCRATCH/'block_mask_check.png');output.parent.mkdir(parents=True,exist_ok=True)
    montage.save(output)
    values=[r['iou'] for r in records if r['iou'] is not None]
    report=dict(median_iou=float(np.median(values)),frames=len(records),nonempty_frames=len(values),
                skipped_empty_colour_frames=empty_controls,
                control='saturation > .35, brightness > .15; analytic union excludes grey blocks',records=records)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    result=check_masks()
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))
    if result['median_iou']<.6: raise SystemExit('Projection control failed: median IoU < .6')
