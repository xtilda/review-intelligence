import csv,io,json
from pathlib import Path
from fastapi import FastAPI,UploadFile,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from app.engine import Sentiment,analyze,ROOT
app=FastAPI(title='Review Intelligence',version='1.0.0');model=Sentiment()
class Reviews(BaseModel):
    texts:list[str]=Field(min_length=1,max_length=1000)
def validate(texts):
    if any(not t.strip() or len(t)>4000 for t in texts):raise HTTPException(422,'Each review must contain 1–4,000 characters.')
    return [t.strip() for t in texts]
@app.get('/')
def home():return FileResponse(ROOT/'app/static/index.html')
@app.get('/health')
def health():return {'status':'ok'}
@app.get('/api/model')
def card():return json.loads((ROOT/'evaluation/report.json').read_text())
@app.post('/api/analyze')
def run(body:Reviews):return analyze(validate(body.texts),model)
@app.post('/api/upload')
async def upload(file:UploadFile):
    try:
        raw=await file.read(2*1024*1024+1)
        if len(raw)>2*1024*1024:raise HTTPException(413,'CSV limit: 2 MB.')
        try:
            reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
            if not reader.fieldnames or 'text' not in reader.fieldnames:raise HTTPException(422,'CSV must have a text column.')
            texts=[]
            for row in reader:
                texts.append(row.get('text') or '')
                if len(texts)>1000:raise HTTPException(422,'Limit: 1,000 reviews.')
            if not texts:raise HTTPException(422,'CSV contains no reviews.')
        except (UnicodeDecodeError,csv.Error) as e:raise HTTPException(422,'Use a valid UTF-8 CSV.') from e
        return analyze(validate(texts),model)
    finally:await file.close()
