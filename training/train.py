"""Reproducible baseline; same text cannot occur across split boundaries."""
import csv,io,json,zipfile,urllib.request,hashlib,re
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
ROOT=Path(__file__).resolve().parents[1]
URL='https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--archive');args=p.parse_args()
    blob=Path(args.archive).read_bytes() if args.archive else urllib.request.urlopen(URL,timeout=60).read()
    rows={};conflicts=set()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('_labelled.txt') or '__MACOSX' in name:continue
            pending=[]
            for line in z.read(name).decode('utf-8-sig').splitlines():
                pending.append(line)
                if not re.search(r'\t[01]$',line):continue
                text,label=' '.join(pending).rsplit('\t',1);pending=[]
                key=' '.join(text.lower().split())
                if key in rows and rows[key][1]!=int(label):conflicts.add(key)
                rows[key]=(text,int(label),Path(name).stem)
            if pending:raise ValueError('Unterminated labelled sentence')
    data=[v for k,v in rows.items() if k not in conflicts]
    train,test=train_test_split(data,test_size=.2,random_state=42,stratify=[x[1] for x in data])
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True)
    x=vectorizer.fit_transform([r[0] for r in train]);model=LogisticRegression(C=1,max_iter=1000,random_state=42).fit(x,[r[1] for r in train])
    xt=vectorizer.transform([r[0] for r in test]);pred=model.predict(xt);y=[r[1] for r in test]
    artifact={'schema':1,'vocabulary':vectorizer.vocabulary_,'idf':vectorizer.idf_.tolist(),'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),'classes':['negative','positive'],'token_pattern':vectorizer.token_pattern,'ngram_range':[1,2],'sublinear_tf':True}
    (ROOT/'models/sentiment.json').write_text(json.dumps(artifact))
    report={'dataset':'UCI Sentiment Labelled Sentences','url':URL,'license':'CC BY 4.0','archive_sha256':hashlib.sha256(blob).hexdigest(),'seed':42,'unique_rows':len(data),'conflicting_texts_removed':len(conflicts),'train_rows':len(train),'test_rows':len(test),'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'confusion_matrix':confusion_matrix(y,pred).tolist(),'classification_report':classification_report(y,pred,output_dict=True),'per_domain':{},'limitations':['English only; balanced short sentences, not production review distribution.','Mixed-domain random split; not a domain-held-out benchmark.','Model probabilities are not calibrated.','Aspect tags use keyword rules, not a trained topic model.']}
    for domain in sorted(set(r[2] for r in test)):
        mask=[i for i,r in enumerate(test) if r[2]==domain];report['per_domain'][domain]={'rows':len(mask),'accuracy':accuracy_score([y[i] for i in mask],[pred[i] for i in mask])}
    (ROOT/'evaluation/report.json').write_text(json.dumps(report,indent=2))
    (ROOT/'evaluation/errors.json').write_text(json.dumps([{'text':r[0],'domain':r[2],'actual':artifact['classes'][label],'predicted':artifact['classes'][int(pr)]} for r,label,pr in zip(test,y,pred) if label!=pr],indent=2))
    # Independent pure-Python serving parity, including OOV and repeated words.
    import sys;sys.path.insert(0,str(ROOT))
    from app.engine import Sentiment
    serving=Sentiment()
    probes=[r[0] for r in test]+['zzqqxxunknown','great great great product']
    expected=model.predict_proba(vectorizer.transform(probes))[:,1]
    delta=max(abs(serving.predict(t)['positive_probability']-float(v)) for t,v in zip(probes,expected))
    assert delta<1e-10,delta
    report['serving_parity_max_absolute_error']=delta
    (ROOT/'evaluation/report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['train_rows','test_rows','accuracy','macro_f1','serving_parity_max_absolute_error']},indent=2))
if __name__=='__main__':main()
