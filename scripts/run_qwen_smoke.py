from pathlib import Path
from shutil import copy2
from app.db import SessionLocal,init_db
from app.models import Batch,MainImage,MatchedFile
from app.services.qwen_production import run_qwen_batch
from datetime import datetime,timezone
root=Path('data/qwen_smoke');main=root/'主图';cat=root/'白底模特图';main.mkdir(parents=True,exist_ok=True);cat.mkdir(parents=True,exist_ok=True)
src=Path(r'C:\Users\Woda\Desktop\9.29');
for i in [1,2,3]: copy2(src/'主图'/f'成人睡裤{i}.jpg',main/f'成人睡裤{i}.jpg')
copy2(src/'白底模特图'/'afaf.png',cat/'afaf.png')
init_db()
with SessionLocal() as db:
 b=Batch(root_path=str(root.resolve()),product_category='smoke',status='SCANNED',match_engine='QWEN',matching_model='QWEN3-VL-8B-INSTRUCT');db.add(b);db.flush()
 for i in [1,2,3]: db.add(MainImage(batch_id=b.id,basename=f'成人睡裤{i}',filename=f'成人睡裤{i}.jpg',file_path=str((main/f'成人睡裤{i}.jpg').resolve())))
 db.add(MatchedFile(batch_id=b.id,category='白底模特图',original_filename='afaf.png',original_path=str((cat/'afaf.png').resolve()),current_filename='afaf.png',current_path=str((cat/'afaf.png').resolve()),extension='.png'))
 db.commit();bid=b.id
 print('batch',bid)
 result=run_qwen_batch(db,bid);print(result)
 item=db.query(MatchedFile).filter_by(batch_id=bid).one(); print('item',item.status,item.matched_main_id,item.match_method,item.is_conflict,len(item.candidates)); print('details',item.match_details_json[:500])
