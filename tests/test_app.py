from fastapi.testclient import TestClient
from app.main import app
from app.engine import Sentiment
c=TestClient(app)
def test_prediction_and_uncertainty():
    m=Sentiment()
    assert m.predict('Excellent service and great quality.')['sentiment']=='positive'
    assert m.predict('zzqqxxunknown')['sentiment']=='uncertain'
    assert m.predict('battery shipping support price easy')['aspects']==['delivery','quality','support','value','usability']
def test_api_limits_and_analysis():
    assert c.get('/').status_code==200
    assert c.get('/api/model').json()['test_rows']>100
    assert c.post('/api/analyze',json={'texts':[]}).status_code==422
    assert c.post('/api/analyze',json={'texts':[' ']}).status_code==422
    assert c.post('/api/analyze',json={'texts':['a'*4001]}).status_code==422
    assert c.post('/api/analyze',json={'texts':['x']*1001}).status_code==422
    d=c.post('/api/analyze',json={'texts':['Great service.','Terrible product.']}).json()
    assert d['total']==sum(d['counts'].values())==2
    assert len(d['rows'])==2

def test_csv():
    assert c.post('/api/upload',files={'file':('r.csv',b'text\n"Great, excellent quality"\n')}).json()['total']==1
    assert c.post('/api/upload',files={'file':('r.csv',b'other\nx\n')}).status_code==422
    assert c.post('/api/upload',files={'file':('r.csv',b'text\n')}).status_code==422
    assert c.post('/api/upload',files={'file':('r.csv',b'\xff')}).status_code==422
    assert c.post('/api/upload',files={'file':('r.csv',b'x'*(2*1024*1024+1))}).status_code==413
