"""Mantém quatro casas decimais, rejeita resultado zero e preserva a fonte alternativa."""
import math
import pytest
from banco_agil.exchange import ExchangeService,ExchangeError,_valid_rate


@pytest.mark.parametrize('value',['0.00001',0.00001,0,-1,'NaN','Infinity','-Infinity','invalid',None])
def test_invalid_or_unrepresentable_rate_is_not_published(value):
    with pytest.raises((ValueError,TypeError)):_valid_rate(value)


@pytest.mark.parametrize('value',[5.43219,5.5,0.0001,0.00006,123.456789])
def test_normal_rate_keeps_existing_four_decimal_rounding(value):
    assert _valid_rate(value)==round(value,4)
    assert math.isfinite(_valid_rate(value)) and _valid_rate(value)>0


class Response:
    def __init__(self,payload):self.payload=payload
    def json(self):return self.payload
    def raise_for_status(self):pass


def test_tiny_primary_rate_uses_valid_secondary():
    calls=[]
    def get(url,**kwargs):
        calls.append(url)
        if len(calls)==1:return Response({'USDBRL':{'bid':'0.00001'}})
        return Response({'rates':{'BRL':5.123456},'date':'2026-10-20'})
    result=ExchangeService(http_get=get).get_rate('USD')
    assert len(calls)==2 and result['fonte']=='Frankfurter (BCE)'
    assert result['moeda']=='USD' and result['valor_em_brl']==5.1235


def test_both_sources_round_to_zero_return_safe_error():
    calls=[]
    def get(url,**kwargs):
        calls.append(url)
        return Response({'USDBRL':{'bid':'0.00001'}} if len(calls)==1 else {'rates':{'BRL':0.00001},'date':'2026-10-20'})
    with pytest.raises(ExchangeError):ExchangeService(http_get=get).get_rate('USD')
    assert len(calls)==2
