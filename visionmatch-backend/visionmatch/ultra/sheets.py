from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..provider import file_hash

LEVELS = {'standard':480, 'high':640, 'ultra':700}
VERSION = 'ultra-sheet-v1'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def save_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)

def dimensions(count, cell=700, columns=None, rows=None):
    if not 1<=count<=50 or cell not in LEVELS.values(): raise ValueError('count must be 1..50, cell 480/640/700')
    columns=columns or (7 if count==50 else math.ceil(math.sqrt(count)))
    rows=rows or math.ceil(count/columns)
    if columns*rows<count or columns<1 or rows<1: raise ValueError('insufficient grid')
    width,height=columns*cell,rows*cell
    patches=math.ceil(width/32)*math.ceil(height/32)
    if patches>30000 or max(width,height)>65535: raise ValueError('raw patch/dimension preflight limit exceeded')
    return {'count':count,'columns':columns,'rows':rows,'cell':cell,'width':width,'height':height,'raw_patches_32':patches,'limit':30000,'effective_model_patches':'UNKNOWN'}

def scan(folder, role):
    folder=Path(folder).resolve()
    files=[p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
    if not 1<=len(files)<=50: raise ValueError(f'{folder}: requires 1..50 images, got {len(files)}')
    result=[]
    for p in files:
        with Image.open(p) as im:
            size=im.size; mode=im.mode; profile=bool(im.info.get('icc_profile')); im.verify()
        result.append({'path':str(p),'name':p.name,'sha256':file_hash(p),'size':list(size),'mode':mode,'icc_profile_present':profile})
    if len({x['sha256'] for x in result})!=len(result): raise ValueError('duplicate image content; resolve explicitly before preparing')
    result.sort(key=lambda x:digest([role,x['sha256']]))
    for i,item in enumerate(result,1): item['id']=f'{role}{i:03}'
    return result

def make_sheet(images, cache_dir, *, cell=700, columns=None, rows=None):
    start=time.perf_counter()
    dims=dimensions(len(images),cell,columns,rows)  # Always guard BEFORE allocating.
    spec={'version':VERSION,'dimensions':dims,'images':[(x['id'],x['sha256']) for x in images]}
    for item in images:
        if file_hash(Path(item['path']))!=item['sha256']:raise ValueError('source changed before cache lookup: '+item['id'])
    key=digest(spec); root=Path(cache_dir)/key; root.mkdir(parents=True,exist_ok=True)
    png=root/'contact_sheet.png'; meta=root/'layout.json'
    if png.exists() and meta.exists():
        known=json.loads(meta.read_text(encoding='utf-8'))
        if file_hash(png)==known['png_sha256']:
            return {**known,'path':str(png),'cache_hit':True,'elapsed_seconds':round(time.perf_counter()-start,3)}
    canvas=Image.new('RGB',(dims['width'],dims['height']),'#eeeeee'); draw=ImageDraw.Draw(canvas)
    label=max(48,round(cell*.1)); margin=12
    font=ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf',round(label*.66)) if Path(r'C:\Windows\Fonts\arialbd.ttf').exists() else ImageFont.load_default()
    tiles=[]
    for i,item in enumerate(images):
        if file_hash(Path(item['path']))!=item['sha256']: raise ValueError('source changed: '+item['id'])
        x,y=(i%dims['columns'])*cell,(i//dims['columns'])*cell
        draw.rectangle((x+2,y+2,x+cell-3,y+cell-3),fill='white',outline='#708090',width=2)
        draw.rectangle((x+3,y+3,x+cell-4,y+label),fill='#193454')
        draw.text((x+12,y+4),item['id'],font=font,fill='white')
        with Image.open(item['path']) as source:
            source=ImageOps.exif_transpose(source)
            if 'A' in source.getbands():
                rgba=source.convert('RGBA'); bg=Image.new('RGBA',rgba.size,'white'); bg.alpha_composite(rgba); source=bg.convert('RGB')
            else: source=source.convert('RGB')
            w,h=source.size; source.thumbnail((cell-2*margin,cell-label-2*margin),Image.Resampling.LANCZOS)
            px=x+(cell-source.width)//2;py=y+label+margin+(cell-label-2*margin-source.height)//2
            canvas.paste(source,(px,py))
            if abs(source.width*h-source.height*w)>max(w,h): raise ValueError('aspect ratio check failed')
            tiles.append({'id':item['id'],'cell':[x,y,cell,cell],'bounds':[px,py,source.width,source.height],'oriented_original_size':[w,h],'rendered_min_dimension':min(source.size),'needs_original_review':min(source.size)<240})
    canvas.save(png,format='PNG')
    known={**spec,'cache_key':key,'png_sha256':file_hash(png),'tiles':tiles,'color_note':'EXIF orientation and aspect-preserving resampling only; no color enhancement. Embedded source ICC profiles are not carried per-tile; non-sRGB source profiles require separate visual validation.'}
    save_json(meta,known)
    return {**known,'path':str(png),'cache_hit':False,'elapsed_seconds':round(time.perf_counter()-start,3)}
