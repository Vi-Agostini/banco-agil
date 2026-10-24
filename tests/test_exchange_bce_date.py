"""A data do BCE deve existir no calendário e seguir o formato YYYY-MM-DD."""
import pytest
import requests
from banco_agil.exchange import ExchangeService,ExchangeError


class Response:
    def __init__(self,payload):self.payload=payload
    def json(self):return self.payload
    def raise_for_status(self):pass


@pytest.mark.parametrize('value',['2026-10-20','2024-02-29','2026-01-01'])
def test_valid_bce_date_is_preserved_without_invented_time(value):
    result=ExchangeService(http_get=lambda *a,**k:Response({'rates':{'BRL':5.123456},'date':value}))._frankfurter('USD')
    assert result['atualizado_em']==value
    assert result['valor_em_brl']==5.1235 and result['fonte']=='Frankfurter (BCE)'


@pytest.mark.parametrize('value',['not-a-date','2026-02-29','2026-02-30','2026-13-01','2026-00-01',
                                  '2026-01-00','0000-01-01','',None,123,[],{},
                                  '20/10/2026','20261020','2026-W43-2','2026-1-01',
                                  '2026-10-20T12:00:00Z',' 2026-10-20','2026-10-20 '])
def test_invalid_metadata_cannot_publish_quote(value):
    def get(url,**kwargs):
        if 'awesomeapi' in url:raise requests.HTTPError('local primary unavailable')
        return Response({'rates':{'BRL':5.123456},'date':value})
    with pytest.raises(ExchangeError):ExchangeService(http_get=get).get_rate('USD')


def test_missing_date_cannot_publish_quote():
    def get(url,**kwargs):
        if 'awesomeapi' in url:raise requests.Timeout('local timeout')
        return Response({'rates':{'BRL':5.123456}})
    with pytest.raises(ExchangeError):ExchangeService(http_get=get).get_rate('USD')


def test_valid_secondary_still_serves_as_fallback():
    calls=[]
    def get(url,**kwargs):
        calls.append(url)
        if len(calls)==1:raise requests.HTTPError('local primary unavailable')
        return Response({'rates':{'BRL':5.123456},'date':'2026-10-20'})
    result=ExchangeService(http_get=get).get_rate('USD')
    assert len(calls)==2 and result['atualizado_em']=='2026-10-20'
    assert result['valor_em_brl']==5.1235


def test_valid_primary_is_not_affected_by_secondary_date_validation():
    calls=[]
    def get(url,**kwargs):
        calls.append(url)
        return Response({'USDBRL':{'bid':'5.43219','create_date':'2026-10-19 12:30:45'}})
    result=ExchangeService(http_get=get).get_rate('USD')
    assert len(calls)==1 and result['fonte']=='AwesomeAPI' and result['valor_em_brl']==5.4322
    assert result['atualizado_em']=='19/10/2026 às 12:30:45 (horário de São Paulo)'
