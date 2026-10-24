"""Separa a espera de capacidade do tempo HTTP, com relógio e rede simulados."""
import json
import logging

import httpx2 as httpx
from openai import OpenAI
import pytest

from banco_agil.agents import AGENTS
from banco_agil.llm import call_llm
from banco_agil.session import Session
from banco_agil.rate_limit import GovernedClient, RateLimiter
from fakes import Clock
from structured_fakes import message


@pytest.mark.parametrize('outcome',['success','400','blocked'])
def test_timings_are_per_attempt_and_do_not_confuse_queue_with_http(outcome,caplog):
    clock=Clock()
    bucket=RateLimiter(30,30000,clock=clock,sleep=clock.sleep)
    bucket.defer(15)
    sent=[]
    def sink(request):
        sent.append(json.loads(request.content))
        clock.now+=2
        if outcome=='400':
            return httpx.Response(400,json={'error':{'code':'invalid_request','param':'response_format'}})
        return httpx.Response(200,json={'id':'local','object':'chat.completion','created':0,'model':'fake',
            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'resposta':message('Olá!')})}}]})
    with caplog.at_level(logging.INFO,logger='banco_agil'):
        with OpenAI(api_key='local',base_url='http://memory.invalid/v1',max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(sink))) as sdk:
            result=call_llm(GovernedClient(sdk,bucket,max_wait=10 if outcome=='blocked' else 60),
                'fake',AGENTS['triagem'],Session(),llm_mode='structured')
    event=next(json.loads(r.message) for r in caplog.records if r.message.startswith('{'))
    assert event['admission_wait_ms']==(0 if outcome=='blocked' else 15000)
    assert event['http_ms']==(0 if outcome=='blocked' else 2000)
    assert event['http_attempted']==(outcome!='blocked')
    assert len(sent)==(0 if outcome=='blocked' else 1)
    assert (result is not None)==(outcome=='success')
