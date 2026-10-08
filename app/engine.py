import json,math,re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ASPECTS={'delivery':['delivery','shipping','arrived','package','packaging'], 'quality':['broken','quality','durable','defective','battery','sound','screen'], 'support':['support','service','refund','helpful','staff'], 'value':['price','expensive','cheap','money','value'], 'usability':['easy','difficult','setup','install','interface','comfortable']}
class Sentiment:
    def __init__(self):
        self.model=json.loads((ROOT/'models/sentiment.json').read_text())
    def predict(self,text):
        m=self.model;words=re.findall(m['token_pattern'],text.lower());grams=words+[' '.join(words[i:i+2]) for i in range(len(words)-1)]
        counts=Counter(g for g in grams if g in m['vocabulary']);features={m['vocabulary'][g]:(1+math.log(n))*m['idf'][m['vocabulary'][g]] for g,n in counts.items()}
        norm=math.sqrt(sum(v*v for v in features.values())) or 1
        logit=m['intercept']+sum(m['coef'][i]*v/norm for i,v in features.items())
        p=1/(1+math.exp(-max(-700,min(700,logit))));confidence=max(p,1-p)
        label=('positive' if p>=.5 else 'negative') if features and confidence>=.6 else 'uncertain'
        return {'sentiment':label,'positive_probability':p,'confidence':confidence,'matched_features':len(features),'aspects':[a for a,keys in ASPECTS.items() if set(words)&set(keys)]}

def analyze(texts,model):
    rows=[dict(id=i+1,text=t,**model.predict(t)) for i,t in enumerate(texts)]
    counts=Counter(r['sentiment'] for r in rows)
    aspects=[{'name':a,'mentions':sum(a in r['aspects'] for r in rows),'negative':sum(a in r['aspects'] and r['sentiment']=='negative' for r in rows)} for a in ASPECTS]
    return {'total':len(rows),'counts':{s:counts[s] for s in ['positive','negative','uncertain']},'aspects':sorted(aspects,key=lambda a:(a['negative'],a['mentions']),reverse=True),'rows':rows}
