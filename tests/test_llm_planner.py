import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from types import SimpleNamespace
import pytest
from arena_rl.commands import Command,SKILLS
from arena_rl.llm_planner import DeepSeekPlanner,Decision,PlannerError,tools_for,apply_decision,strict_json


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','fake-test-credential')
    holder={'requests':[],'status':200,'headers':{}}
    holder['response']={'id':'test-response','model':'deepseek-flash','usage':{'total_tokens':12},
       'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[{'id':'call-1','type':'function',
          'function':{'name':'resupply','arguments':json.dumps({'ammo_target':100,'hp_target_fraction':.9,'budget_limit':150,'timeout_s':60,'intent':'reload'})}}]}}]}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            holder['requests'].append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(holder['status'])
            for key,value in holder['headers'].items():self.send_header(key,value)
            self.end_headers();self.wfile.write(json.dumps(holder['response']).encode())
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    holder['url']=f'http://127.0.0.1:{server.server_port}'
    try:yield holder
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_wire_call_uses_exact_model_and_host_owned_identity(provider):
    client=DeepSeekPlanner(base_url=provider['url'])
    d=client.decide({'tick':42},['resupply'],{'season':'2027'})
    body=provider['requests'][0]
    assert body['model']=='deepseek-flash' and body['thinking']=={'type':'disabled'}
    assert body['parallel_tool_calls'] is False and body['tool_choice']=='required'
    assert 'fake-test-credential' not in json.dumps(body)
    assert [t['function']['name'] for t in body['tools']]==['resupply','continue_current','cancel_current']
    command=d.command('host-id')
    assert command['basis_tick']==42 and command['id']=='host-id' and command['skill']=='resupply'
    assert d.response_id=='test-response' and d.usage['total_tokens']==12


@pytest.mark.parametrize('corruption',['multiple','unknown','actuators','nan','incomplete','null_point'])
def test_invalid_model_calls_are_rejected(provider,corruption):
    message=provider['response']['choices'][0]
    call=message['message']['tool_calls'][0]
    if corruption=='multiple':message['message']['tool_calls'].append(call)
    elif corruption=='unknown':call['function']['name']='teleport'
    elif corruption=='actuators':
        args=json.loads(call['function']['arguments']);args['slot']=4;call['function']['arguments']=json.dumps(args)
    elif corruption=='nan':call['function']['arguments']='{"ammo_target":NaN}'
    elif corruption=='incomplete':message['finish_reason']='length'
    else:
        call['function']={'name':'navigate','arguments':json.dumps({'point':None,'radius_m':None,'timeout_s':None,'low_hp_fraction':None,'intent':'move'})}
    with pytest.raises(PlannerError,match='invalid_tool_call'):
        DeepSeekPlanner(base_url=provider['url']).decide({'tick':0},['resupply','navigate'],{})


def test_provider_errors_do_not_echo_credentials(provider):
    provider.update(status=401,response={'error':'fake-test-credential and private provider text'})
    with pytest.raises(PlannerError) as caught:DeepSeekPlanner(base_url=provider['url']).decide({'tick':0},['resupply'],{})
    assert caught.value.status==401 and not caught.value.retryable
    assert 'credential' not in str(caught.value) and 'private' not in str(caught.value)


def test_redirect_never_forwards_authorization(provider):
    provider.update(status=302,headers={'Location':'https://unrelated.invalid/steal'})
    with pytest.raises(PlannerError):DeepSeekPlanner(base_url=provider['url']).decide({'tick':0},['resupply'],{})
    assert len(provider['requests'])==1


def test_no_key_or_insecure_remote_endpoint(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False);monkeypatch.delenv('DeepSeek_API_Key',raising=False)
    with pytest.raises(PlannerError,match='missing_api_key'):DeepSeekPlanner()
    with pytest.raises(ValueError,match='HTTPS'):DeepSeekPlanner(base_url='http://remote.invalid')


def test_tool_surface_has_no_actuator_or_episode_operations():
    tools=tools_for(SKILLS)
    for t in tools:
        f=t['function'];p=f['parameters']
        assert p['additionalProperties'] is False and set(p['required'])==set(p['properties'])
        assert not set(p['properties']) & {'slot','team','move','fire','ammo','hp','id','basis_tick','reset'}
    assert len(tools)==14
    with pytest.raises(ValueError):strict_json('{"x":1,"x":2}')


class Session:
    def __init__(self):
        self.calls=0;self.ex=SimpleNamespace(status='running',command=Command.parse({'id':'old','skill':'navigate','point':[5,5]}))
        self.batch=SimpleNamespace(executors=[self.ex])
    def observe(self,i=0):return {'tick':100,'feedback':{'status':self.ex.status}}
    def command(self,command,i=0):self.calls+=1;return {'accepted':True}
    def cancel(self,i=0):self.calls+=1;return {'status':'cancelled'}


def test_stale_and_identical_decisions_preserve_running_goal():
    session=Session();args={'point':[5,5],'radius_m':None,'timeout_s':None,'intent':'move'}
    d=Decision('navigate',args,'deepseek-flash','deepseek-flash','r',{},1.,100,'c')
    result=apply_decision(session,d,0,'new');assert result['action']=='same_goal_preserved' and session.calls==0
    d.basis_tick=0
    with pytest.raises(ValueError,match='Stale'):apply_decision(session,d,0,'late',1.)
    assert session.calls==0
    d.basis_tick=101
    with pytest.raises(ValueError,match='Stale'):apply_decision(session,d,0,'future')


def test_coordinate_tools_use_bounded_scalars_and_strict_provider_schema():
    tool=tools_for(['navigate'],{'width':28.,'height':15.,'margin':.3})[0]['function']
    assert tool['strict'] is True
    point=tool['parameters']['properties']['point']
    assert point['type']=='object' and set(point['required'])=={'x','y'} and point['additionalProperties'] is False
    assert point['properties']['x']['maximum']==27.7 and point['properties']['y']['minimum']==.3


def test_provider_scalar_coordinate_roundtrip(provider):
    call=provider['response']['choices'][0]['message']['tool_calls'][0]
    call['function']={'name':'navigate','arguments':json.dumps({'point':{'x':4.5,'y':5.15},'radius_m':None,'timeout_s':None,'low_hp_fraction':0.,'intent':'withdraw'})}
    decision=DeepSeekPlanner(base_url=provider['url']).decide({'tick':0,'robot':{'radius':.3}},['navigate'],{'field':{'width':28.,'height':15.}})
    assert decision.command('scalar')['point']==[4.5,5.15]
    call['function']['arguments']=json.dumps({'point':{'x':29.,'y':5.},'radius_m':None,'timeout_s':None,'low_hp_fraction':0.,'intent':'bad point'})
    with pytest.raises(PlannerError,match='invalid_tool_call'):
        DeepSeekPlanner(base_url=provider['url']).decide({'tick':0},['navigate'],{'field':{'width':28.,'height':15.}})


def test_catalog_point_is_resolved_and_unknown_id_is_rejected(provider):
    call=provider['response']['choices'][0]['message']['tool_calls'][0]
    args={'point':'staging','radius_m':None,'timeout_s':None,'low_hp_fraction':None,'intent':'stage'}
    call['function']={'name':'navigate','arguments':json.dumps(args)}
    context={'field':{'width':28.,'height':15.},'safe_waypoints':{'staging':[4.5,5.15]}}
    d=DeepSeekPlanner(base_url=provider['url']).decide({'tick':0},['navigate'],context)
    assert d.command('catalog')['point']==[4.5,5.15]
    schema=provider['requests'][-1]['tools'][0]['function']['parameters']['properties']['point']
    assert schema=={'type':'string','enum':['staging']}
    args['point']='unregistered';call['function']['arguments']=json.dumps(args)
    with pytest.raises(PlannerError):DeepSeekPlanner(base_url=provider['url']).decide({'tick':0},['navigate'],context)
