"""Documentation figures in Chinese and English, rendered from committed data.

Usage: uv run --with matplotlib python examples/make_figures.py [--lang zh en] [--out docs/images]
Sources: checkpoints/ground and checkpoints/serial305 (training/evaluation), evaluations/team-matches-2026-10-19
(match reports, LLM decision logs, downsampled match traces) and Arena's public 2027 map geometry.
"""
import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle,Polygon,Rectangle
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
MATCHES=ROOT/'evaluations'/'team-matches-2026-10-19'
RED,BLUE='#d64545','#3567c9'
POLICY,EXPERT,RANDOM='#2f8f62','#8c8c8c','#d9a441'
FAMILIES=('navigation','combat','defense','support')
MODELS=(3,4,7,301,302,303,304,305,2731,2732,2711,2712,2713,2771,2772)
NAMES={3:'W3',4:'I4',7:'I7',301:'I-T',302:'I-S',303:'I-O',304:'I-W',305:'I-L',2731:'I27-W',2732:'I27-T',2711:'HV-A',2712:'HV-D',2713:'HV-O',2771:'S27-O',2772:'S27-L'}
DRIVE={3:'legged',4:'mecanum',7:'mecanum',301:'mecanum',302:'swerve',303:'omni',304:'legged',305:'legged',2731:'legged',2732:'mecanum',2711:'mecanum',2712:'swerve',2713:'omni',2771:'omni',2772:'legged'}

T={'zh':{
 'family':{'navigation':'导航','combat':'作战','defense':'防守','support':'保障'},
 'role':{'infantry':'步兵','heavy':'重型','sentry':'哨兵','aerial':'空中'},
 'drive':{'legged':'轮足','mecanum':'麦轮','omni':'全向','swerve':'舵轮'},
 'modes':{'policy':'RL 策略','expert':'专家控制器','random':'随机动作'},
 'success':'成功率','samples':'PPO 样本（千）','field_title':'RMUC 2027 场地与关键目标（红方视角）',
 'zones':{'base':'基地','outpost':'前哨站','fortress':'堡垒卡','central':'中央高地卡','supply':'补给区','high_point':'制高点',
          'workbench':'工作台','assembly':'装配区','hangar':'机库','rune':'能量机关','bumps':'公路起伏路段'},
 'legend_platform':'高地/平台','legend_ramp':'坡道','legend_bumps':'起伏路段','legend_wall':'墙体/障碍',
 'traj_title':'LLM 指挥 + RL 执行的一局完整对局（种子 {seed}，{winner}胜，420 s）','start':'起点','drones':'空中机器人航迹',
 'winner':{'red':'红方','blue':'蓝方'},
 'timeline_title':'LLM 对每台红方机器人的高层指令（种子 {seed}）','time':'比赛时间 (s)',
 'control':'控制度','outpost_hp':'前哨站血量','red_control':'红方控制度','blue_control':'蓝方控制度','red_outpost':'红方前哨站','blue_outpost':'蓝方前哨站',
 'fallback':'规则兜底 / 空闲补令',
 'objectives':{'resupply':'补给','capture_enemy_central':'夺敌方中央卡','hold_own_central':'守己方中央卡','occupy_enemy_fortress':'占敌方堡垒',
   'hold_own_fortress':'守己方堡垒','attack_outpost':'打前哨站','push_base':'推家','defend_base':'守家','pursue_enemy':'追击敌方',
   'engage_enemy':'交战','retreat':'撤退','escort':'护送','patrol':'巡逻','goto':'前往','hold_position':'原地守卫','install_launcher':'安装发射机构',
   'drone_strike_outpost':'空中撞击前哨','drone_strike_base':'空中撞击基地','drone_recall':'召回空中机器人'},
 'curves_title':'通用策略训练曲线（采样窗口成功率，带探索噪声）','all_models':'全部型号','heavy_override':'重型覆盖策略',
 'aerial_curve':'空中俯冲策略','rotating':'旋转前哨','static':'停转前哨',
 'heat_title':'每型号评测成功率（独立种子，每格 8 回合）','heat_left':'RL 策略','heat_right':'RL 策略 − 专家',
 'eval_title':'评测汇总：RL 策略 vs 专家 vs 随机','strikes':'每回合有效撞击','first_hit':'首次命中中位数 {v:.1f} s',
 'aerial_title':'空中机器人（36 回合）',
 'tp_title':'本机训练吞吐优化（Apple M5，10 核）','workers':'采样进程数','sps':'样本/秒','sync':'SB3 SubprocVecEnv（每步同步）','async':'解耦采样进程',
 'before':'优化前','after':'优化后','bottlenecks':['模拟单步耗时 (ms)\n平底盘压起伏块','关闭模拟器 (s)\n换阵容时','推家预热耗时 (s)\n辅助机器人直接上卡','作战训练吞吐 (样本/秒)\n异步+缓存'],
 'team_title':'完整 420 s 对局（蓝方均为规则队伍 doctrine + 专家；胜/负为红方结果）','configs':['规则+专家\n(镜像基线)','规则+RL','LLM+RL'],
 'damage':'攻击伤害','match':'局','win':'胜','loss':'负',
 's305_title':'串联腿 305 各策略族独立评测成功率','s305':{'combat':'作战','defense':'防守','support':'保障','mobility':'机动/自扶正','flight':'飞坡','flight_heldout':'飞坡(扩展)','navigation_heldout':'导航(留出)','navigation_composed':'导航(组合)'},
 'arch':['顶层 LLM 指挥官','DeepSeek 函数调用 · 每 6 s 或事件触发 · 异步','Harness','场景摘要 · 规则状态 · 可行性过滤 · 指令编译','规则指挥官兜底 · 空闲补令 · 过期拒绝 · 日志',
         'TeamController','每台地面机器人一个 TacticalExecutor','空中机库控制器','通用地面 RL 策略库','导航 · 作战 · 防守 · 保障 · 重型覆盖','空中俯冲 RL 策略',
         'Arena 模拟器（Rust 物理 + 2027 裁判）','重型 · 步兵×2 · 空中 · 哨兵','issue_orders','战术指令','10 Hz 动作','观测 / 执行反馈',
         '离线训练','课程（真实预热）→ 异步采样进程 → 专家示范行为克隆 → PPO（CPU/CUDA/MPS）','策略检查点','俯冲撞击旋转 / 停转前哨'],
},'en':{
 'family':{'navigation':'Navigation','combat':'Combat','defense':'Defense','support':'Support'},
 'role':{'infantry':'infantry','heavy':'heavy','sentry':'sentry','aerial':'aerial'},
 'drive':{'legged':'wheel-legged','mecanum':'mecanum','omni':'omni','swerve':'swerve'},
 'modes':{'policy':'RL policy','expert':'expert controller','random':'random actions'},
 'success':'success rate','samples':'PPO samples (k)','field_title':'RMUC 2027 field and key objectives (red perspective)',
 'zones':{'base':'base','outpost':'outpost','fortress':'fortress card','central':'central card','supply':'supply','high_point':'high point',
          'workbench':'workbench','assembly':'assembly','hangar':'hangar','rune':'rune','bumps':'road bumps'},
 'legend_platform':'highland / platform','legend_ramp':'ramp','legend_bumps':'road bumps','legend_wall':'wall / obstacle',
 'traj_title':'A full match: LLM command + RL execution (seed {seed}, {winner} wins, 420 s)','start':'start','drones':'aerial robot paths',
 'winner':{'red':'red','blue':'blue'},
 'timeline_title':'LLM objectives for each red robot (seed {seed})','time':'match time (s)',
 'control':'control degree','outpost_hp':'outpost HP','red_control':'red control','blue_control':'blue control','red_outpost':'red outpost','blue_outpost':'blue outpost',
 'fallback':'doctrine fallback / idle safety net',
 'objectives':{'resupply':'resupply','capture_enemy_central':'capture enemy central','hold_own_central':'hold own central','occupy_enemy_fortress':'occupy enemy fortress',
   'hold_own_fortress':'hold own fortress','attack_outpost':'attack outpost','push_base':'push base','defend_base':'defend base','pursue_enemy':'pursue enemy',
   'engage_enemy':'engage enemy','retreat':'retreat','escort':'escort','patrol':'patrol','goto':'goto','hold_position':'hold position','install_launcher':'install launcher',
   'drone_strike_outpost':'drone strike outpost','drone_strike_base':'drone strike base','drone_recall':'drone recall'},
 'curves_title':'Universal policy training curves (rollout-window success, with exploration noise)','all_models':'all models','heavy_override':'heavy override',
 'aerial_curve':'aerial dive policy','rotating':'rotating outpost','static':'stopped outpost',
 'heat_title':'Per-model evaluation success (held-out seeds, 8 episodes per cell)','heat_left':'RL policy','heat_right':'RL policy − expert',
 'eval_title':'Evaluation summary: RL policy vs expert vs random','strikes':'valid rams per episode','first_hit':'median first hit {v:.1f} s',
 'aerial_title':'Aerial robots (36 episodes)',
 'tp_title':'Training throughput fixes on one Mac (Apple M5, 10 cores)','workers':'rollout processes','sps':'samples / s','sync':'SB3 SubprocVecEnv (sync every step)','async':'decoupled workers',
 'before':'before','after':'after','bottlenecks':['sim step (ms)\nflat chassis on bumps','simulator close (s)\nlineup change','base-push warmup (s)\nhelpers start on cards','combat throughput (samples/s)\nasync + caching'],
 'team_title':'Full 420 s matches (blue is always the declared doctrine + expert team; win/loss is the red result)','configs':['doctrine + expert\n(mirror baseline)','doctrine + RL','LLM + RL'],
 'damage':'attack damage','match':'match','win':'win','loss':'loss',
 's305_title':'Serial-leg 305: held-out success by policy family','s305':{'combat':'combat','defense':'defense','support':'support','mobility':'mobility / self-righting','flight':'fly slope','flight_heldout':'fly slope (wider)','navigation_heldout':'navigation (held-out)','navigation_composed':'navigation (composed)'},
 'arch':['Top-level LLM commander','DeepSeek function calling · every 6 s or on events · async','Harness','scene summary · rule status · feasibility filter · order compiler','doctrine fallback · idle safety net · stale rejection · logs',
         'TeamController','one TacticalExecutor per ground robot','aerial hangar controller','universal ground RL policies','navigation · combat · defense · support · heavy overrides','aerial dive RL policy',
         'Arena simulator (Rust physics + 2027 referee)','heavy · 2 infantry · aerial · sentry','issue_orders','tactical commands','10 Hz actions','observations / feedback',
         'offline training','curricula (earned warmups) → async rollout workers → expert BC → PPO (CPU/CUDA/MPS)','checkpoints','dive strikes on rotating / stopped outposts'],
}}


def setup(lang):
    plt.rcParams.update({'figure.dpi':110,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,
                         'axes.unicode_minus':False,'font.size':10,'axes.titleweight':'bold'})
    if lang=='zh':
        for path in ('/System/Library/Fonts/Hiragino Sans GB.ttc','/Library/Fonts/Arial Unicode.ttf'):
            if Path(path).is_file():font_manager.fontManager.addfont(path)
        plt.rcParams['font.family']=['Hiragino Sans GB','Arial Unicode MS','Noto Sans CJK SC','DejaVu Sans']
    else:plt.rcParams['font.family']=['DejaVu Sans']


def load_json(path):
    path=Path(path);data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def arena():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from arena_rl.bridge import configure
    configure()
    from rmuc_arena import sdk
    return sdk.ARENAS_BY_SEASON['2027']


def rect(item,**style):
    angle=item.get('rotation',0.);w,h=item['w'],item['h'];cx,cy=item['x']+w/2,item['y']+h/2
    c,s=math.cos(angle),math.sin(angle);corners=[(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
    return Polygon([(cx+x*c-y*s,cy+x*s+y*c) for x,y in corners],closed=True,**style)


def draw_field(ax,geometry,t,labels=True,faint=False):
    alpha=.45 if faint else 1.
    ax.add_patch(Rectangle((0,0),geometry['width'],geometry['height'],facecolor='#f4f1ea',edgecolor='#444',lw=1.2,zorder=0))
    for item in geometry['terrain']:
        kind=item['kind'];style={'platform':('#d9d2c3',None),'bevel':('#cfc6b2',None),'ramp':('#e8e0cf','///'),'bumps':('#efe2b8','xxx')}.get(kind,('#ddd',None))
        patch=Polygon(item['polygon'],closed=True) if item.get('polygon') else rect(item)
        patch.set(facecolor=style[0],edgecolor='#b9ae96',lw=.5,hatch=style[1],alpha=alpha,zorder=1);ax.add_patch(patch)
    for wall in geometry['obstacles']:
        if 'roof' in wall.get('id',''):continue
        ax.add_patch(rect(wall,facecolor='#5b5b5b',edgecolor='none',alpha=.85*alpha,zorder=2))
    colors={'red':RED,'blue':BLUE,None:'#777'}
    for zone in geometry['zones']:
        if zone['kind'] not in ('base','outpost','fortress','central','supply','high_point','workbench','assembly'):continue
        color=colors.get(zone.get('owner'),'#777')
        patch=Polygon(zone['polygon'],closed=True) if zone.get('polygon') else Circle((zone['x'],zone['y']),zone.get('radius',.4))
        patch.set(facecolor=color,edgecolor=color,alpha=.22*alpha if zone['kind'] in ('supply','high_point','assembly') else .45*alpha,lw=1,zorder=3);ax.add_patch(patch)
    for team,hangar in geometry.get('hangars',{}).items():
        ax.add_patch(Rectangle((hangar['x']-.2,hangar['y']-.2),.4,.4,facecolor=colors[team],edgecolor='k',lw=.6,alpha=alpha,zorder=4))
    rune=geometry.get('rune',{})
    if rune:ax.add_patch(Circle((rune['x'],rune['y']),.35,facecolor='#9b59b6',edgecolor='k',lw=.6,alpha=.7*alpha,zorder=4))
    if labels:
        seen=set()
        for zone in geometry['zones']:
            kind=zone['kind']
            if kind not in ('base','outpost','fortress','central','supply','high_point') or (kind,zone.get('owner')) in seen:continue
            seen.add((kind,zone.get('owner')));team=zone.get('owner')
            dy=.75 if kind in ('fortress','central') else -.75 if kind in ('outpost',) else .0
            ax.annotate(t['zones'][kind],(zone['x'],zone['y']),xytext=(zone['x'],zone['y']+dy+(.0 if team=='red' else .0)),ha='center',va='center',fontsize=8,
                        color=colors.get(team,'#333'),fontweight='bold',zorder=6,bbox=dict(boxstyle='round,pad=.15',fc='white',ec='none',alpha=.75))
        for team,hangar in geometry.get('hangars',{}).items():
            ax.annotate(t['zones']['hangar'],(hangar['x'],hangar['y']),xytext=(hangar['x']+(-.9 if team=='red' else .9),hangar['y']+(.55 if team=='red' else -.55)),ha='center',fontsize=7.5,color=colors[team],zorder=6,
                        bbox=dict(boxstyle='round,pad=.12',fc='white',ec='none',alpha=.7))
        if rune:ax.annotate(t['zones']['rune'],(rune['x'],rune['y']),xytext=(rune['x'],rune['y']-.8),ha='center',fontsize=7.5,color='#6c3483',zorder=6,
                            bbox=dict(boxstyle='round,pad=.12',fc='white',ec='none',alpha=.7))
        for item in geometry['terrain']:
            if item['kind']=='bumps':
                ax.annotate(t['zones']['bumps'],(item['x']+item['w']/2,item['y']+item['h']/2),ha='center',va='center',fontsize=6.5,color='#7a6420',zorder=6)
    ax.set_xlim(-.3,geometry['width']+.3);ax.set_ylim(-.3,geometry['height']+.3);ax.set_aspect('equal');ax.grid(False)
    ax.set_xticks(range(0,29,4));ax.set_yticks(range(0,16,3));ax.tick_params(labelsize=7)


def fig_field(t,out,geometry):
    fig,ax=plt.subplots(figsize=(11,6.4));draw_field(ax,geometry,t)
    handles=[Rectangle((0,0),1,1,fc='#d9d2c3',ec='#b9ae96',label=t['legend_platform']),Rectangle((0,0),1,1,fc='#e8e0cf',ec='#b9ae96',hatch='///',label=t['legend_ramp']),
             Rectangle((0,0),1,1,fc='#efe2b8',ec='#b9ae96',hatch='xxx',label=t['legend_bumps']),Rectangle((0,0),1,1,fc='#5b5b5b',label=t['legend_wall']),
             Rectangle((0,0),1,1,fc=RED,alpha=.45,label=t['winner']['red']),Rectangle((0,0),1,1,fc=BLUE,alpha=.45,label=t['winner']['blue'])]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.07),ncol=6,fontsize=8,frameon=False)
    ax.set_title(t['field_title']);fig.tight_layout();fig.savefig(out/'field-2027.png',bbox_inches='tight');plt.close(fig)


def fig_trajectories(t,out,geometry,trace,seed,winner):
    fig,ax=plt.subplots(figsize=(11,6.6));draw_field(ax,geometry,t,labels=False,faint=True)
    paths=defaultdict(list);drones=defaultdict(list)
    for frame in trace['frames']:
        for r in frame['robots']:
            if r['role']=='aerial':
                for k,d in enumerate(r.get('drones',[])):
                    if d[3]=='flying':drones[(r['id'],k)].append((d[0],d[1]))
            elif r['alive']:paths[r['id']].append((r['x'],r['y']))
            else:paths[r['id']].append((np.nan,np.nan))
    styles={'1':('-',2.2),'3':('-',1.5),'4':('--',1.5),'7':(':',2.)}
    roles={}
    for frame in trace['frames'][:1]:
        for r in frame['robots']:roles[r['id']]=r['role']
    for identity,points in sorted(paths.items(),key=lambda item:(not item[0].startswith('red'),item[0])):
        team,slot=identity.split('-');p=np.array(points);ls,lw=styles.get(slot,('-',1.2))
        ax.plot(p[:,0],p[:,1],ls,color=RED if team=='red' else BLUE,lw=lw,alpha=.85,zorder=5,label=f"{t['winner'][team]} {slot} {t['role'][roles.get(identity,'infantry')]}")
        first=p[~np.isnan(p[:,0])][0];ax.plot(*first,'o',color=RED if team=='red' else BLUE,ms=6,mec='k',mew=.6,zorder=6)
    for (identity,_),points in drones.items():
        p=np.array(points);ax.plot(p[:,0],p[:,1],'-',color=RED if identity.startswith('red') else BLUE,lw=.5,alpha=.35,zorder=4)
    handles,labels=ax.get_legend_handles_labels()
    handles+=[Line2D([],[],color='#555',lw=.6,alpha=.6,label=t['drones']),Line2D([],[],marker='o',ls='',color='#555',mec='k',label=t['start'])]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.06),ncol=5,fontsize=8,frameon=False)
    ax.set_title(t['traj_title'].format(seed=seed,winner=t['winner'][winner]));fig.tight_layout();fig.savefig(out/'match-trajectories.png',bbox_inches='tight');plt.close(fig)


def segments_from_decisions(rows,end_s):
    """Objective in force for each robot: commanded orders hold until the next command for that robot."""
    events=defaultdict(list)
    for row in rows:
        if row['type'] not in ('decision','safety_orders','doctrine_orders'):continue
        tick=row.get('applied_tick',row.get('tick'));source='fallback' if row['type']!='decision' else 'llm'
        for robot,result in row['results'].items():
            if result.get('action')=='commanded':events[robot].append((tick/20.,result['objective'],source))
    segments={}
    for robot,items in events.items():
        items.sort();segments[robot]=[(start,(items[k+1][0] if k+1<len(items) else end_s),objective,source) for k,(start,objective,source) in enumerate(items)]
    return segments


def fig_timeline(t,out,decisions,trace,seed):
    end=trace['frames'][-1]['t'];segments=segments_from_decisions(decisions,end)
    robots=['red-1','red-3','red-4','red-7','red-6'];labels={'red-1':'1','red-3':'3','red-4':'4','red-7':'7','red-6':'6'}
    roles={'red-1':'heavy','red-3':'infantry','red-4':'infantry','red-7':'sentry','red-6':'aerial'}
    used=sorted({s[2] for robot in robots for s in segments.get(robot,[])},key=lambda o:list(T['en']['objectives']).index(o) if o in T['en']['objectives'] else 99)
    cmap=plt.get_cmap('tab20');color={o:cmap(k%20) for k,o in enumerate(used)}
    fig,(ax,bx)=plt.subplots(2,1,figsize=(12,6.8),gridspec_kw={'height_ratios':[3,1.6]},sharex=True)
    for row,robot in enumerate(robots):
        for start,stop,objective,source in segments.get(robot,[]):
            ax.barh(row,stop-start,left=start,height=.62,color=color[objective],edgecolor='white' if source=='llm' else 'k',lw=.4,hatch='////' if source=='fallback' else None)
    ax.set_yticks(range(len(robots)));ax.set_yticklabels([f"{t['winner']['red']} {labels[r]} · {t['role'][roles[r]]}" for r in robots]);ax.invert_yaxis()
    handles=[Rectangle((0,0),1,1,fc=color[o],label=t['objectives'].get(o,o)) for o in used]+[Rectangle((0,0),1,1,fc='white',ec='k',hatch='////',label=t['fallback'])]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=6,fontsize=8,frameon=False)
    fig.suptitle(t['timeline_title'].format(seed=seed),fontweight='bold',y=.995);ax.grid(axis='y',alpha=0)
    times=[f['t'] for f in trace['frames']]
    bx.plot(times,[f['control'].get('red',0) for f in trace['frames']],color=RED,lw=2,label=t['red_control'])
    bx.plot(times,[f['control'].get('blue',0) for f in trace['frames']],color=BLUE,lw=2,label=t['blue_control'])
    bx.set_ylabel(t['control']);cx=bx.twinx();cx.grid(False)
    cx.plot(times,[f['outposts'].get('red',0) for f in trace['frames']],color=RED,lw=1,ls='--',alpha=.7,label=t['red_outpost'])
    cx.plot(times,[f['outposts'].get('blue',0) for f in trace['frames']],color=BLUE,lw=1,ls='--',alpha=.7,label=t['blue_outpost'])
    cx.set_ylabel(t['outpost_hp']);cx.spines['right'].set_visible(True)
    h1,l1=bx.get_legend_handles_labels();h2,l2=cx.get_legend_handles_labels();bx.legend(h1+h2,l1+l2,fontsize=8,ncol=4,loc='upper left',frameon=False)
    bx.set_xlabel(t['time']);bx.set_xlim(0,end);fig.tight_layout(rect=(0,0,1,.885));fig.savefig(out/'llm-timeline.png',bbox_inches='tight');plt.close(fig)


def metrics(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def smooth(values,k=5):
    values=np.asarray(values,dtype=float)
    if len(values)<k:return values
    return np.convolve(values,np.ones(k)/k,mode='valid')


def fig_curves(t,out,bank):
    fig,axes=plt.subplots(2,3,figsize=(13,7));axes=axes.ravel()
    classes={'wheel_legged':'legged','mecanum':'mecanum','omni':'omni','swerve':'swerve'}
    for ax,family in zip(axes,FAMILIES):
        rows=metrics(bank/family/'metrics.jsonl');x=np.array([r['timesteps'] for r in rows])/1000
        y=[r['window']['success_rate'] for r in rows];ax.plot(x[len(x)-len(smooth(y)):],smooth(y),color='k',lw=2.2,label=t['all_models'])
        by=defaultdict(list)
        for r in rows:
            for key,v in r['window']['by_class'].items():by[key].append((r['timesteps']/1000,v['success_rate']))
        for key,points in sorted(by.items()):
            role,drive=key.split('/');p=np.array(points)
            if len(p)<8:continue
            ax.plot(p[len(p)-len(smooth(p[:,1])):,0],smooth(p[:,1]),lw=1,alpha=.75,label=f"{t['role'][role]}/{t['drive'][classes.get(drive,drive)]}")
        heavy=bank/f'{family}-heavy'/'metrics.jsonl'
        if heavy.is_file():
            hr=metrics(heavy);hx=x[-1]+np.array([r['timesteps'] for r in hr])/1000;hy=[r['window']['success_rate'] for r in hr]
            ax.plot(hx[len(hx)-len(smooth(hy)):],smooth(hy),color='#8e44ad',lw=2,ls='--',label=t['heavy_override'])
        ax.set_title(t['family'][family]);ax.set_ylim(0,1.02);ax.set_xlabel(t['samples']);ax.set_ylabel(t['success'])
        ax.legend(fontsize=6.5,ncol=2,frameon=False,loc='lower right')
    ax=axes[4];rows=metrics(bank/'aerial'/'metrics.jsonl')
    for variant,colour in (('aerial-rotating','#16a085'),('aerial-static','#e67e22')):
        p=np.array([(r['timesteps']/1000,r['tasks'][variant]['success_rate']) for r in rows if variant in r['tasks']])
        ax.plot(p[len(p)-len(smooth(p[:,1])):,0],smooth(p[:,1]),color=colour,lw=2,label=t['rotating' if 'rotating' in variant else 'static'])
    ax.set_title(t['aerial_curve']);ax.set_ylim(0,1.02);ax.set_xlabel(t['samples']);ax.set_ylabel(t['success']);ax.legend(fontsize=8,frameon=False)
    ax=axes[5];names=[];steps=[];minutes=[]
    for family in FAMILIES+('combat-heavy','defense-heavy','aerial'):
        meta=json.loads((bank/family/'metadata.json').read_text())
        names.append(t['role']['aerial'] if family=='aerial' else t['family'][family.split('-')[0]]+'·'+t['role']['heavy'] if family.endswith('-heavy') else t['family'][family])
        steps.append(meta['actual_steps']/1000);minutes.append(meta['wall_seconds']/60)
    bars=ax.barh(range(len(names)),steps,color='#5d6d7e');ax.set_yticks(range(len(names)));ax.set_yticklabels(names,fontsize=8);ax.invert_yaxis()
    for bar,m in zip(bars,minutes):ax.text(bar.get_width()+15,bar.get_y()+bar.get_height()/2,f'{m:.0f} min',va='center',fontsize=7.5)
    ax.set_xlabel(t['samples']);ax.set_title('PPO');ax.grid(axis='y',alpha=0)
    fig.suptitle(t['curves_title'],fontweight='bold');fig.tight_layout();fig.savefig(out/'training-curves.png',bbox_inches='tight');plt.close(fig)


def evaluation_table(bank):
    table={}
    for family in FAMILIES:
        report=json.loads((bank/family/'evaluation.json').read_text());models=dict(report['models'])
        heavy=bank/f'{family}-heavy'/'evaluation.json'
        if heavy.is_file():models.update(json.loads(heavy.read_text())['models'])
        table[family]={int(m):{mode:v['success_rate'] for mode,v in row.items() if isinstance(v,dict) and 'success_rate' in v} for m,row in models.items()}
    return table


def fig_heatmap(t,out,bank):
    table=evaluation_table(bank)
    policy=np.array([[table[f].get(m,{}).get('policy',np.nan) for f in FAMILIES] for m in MODELS])
    delta=np.array([[table[f].get(m,{}).get('policy',np.nan)-table[f].get(m,{}).get('expert',np.nan) for f in FAMILIES] for m in MODELS])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,7.4))
    rows=[f"{m} {NAMES[m]} · {t['drive'][DRIVE[m]]}" for m in MODELS]
    for axis,data,cmap,vmin,vmax,title in ((ax,policy,'YlGn',0,1,t['heat_left']),(bx,delta,'RdBu',-.5,.5,t['heat_right'])):
        image=axis.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i,j]):axis.text(j,i,f'{data[i,j]:+.2f}' if axis is bx else f'{data[i,j]:.2f}',ha='center',va='center',fontsize=8,
                                                   color='white' if (axis is ax and data[i,j]>.75) or (axis is bx and abs(data[i,j])>.35) else 'black')
        axis.set_xticks(range(len(FAMILIES)));axis.set_xticklabels([t['family'][f] for f in FAMILIES]);axis.set_yticks(range(len(MODELS)))
        axis.set_yticklabels(rows if axis is ax else [],fontsize=8);axis.set_title(title);axis.grid(False)
        for boundary in (9.5,12.5):axis.axhline(boundary,color='k',lw=1.2)
        fig.colorbar(image,ax=axis,fraction=.046,pad=.03)
    fig.suptitle(t['heat_title'],fontweight='bold');fig.tight_layout();fig.savefig(out/'eval-heatmap.png',bbox_inches='tight');plt.close(fig)


def fig_eval_summary(t,out,bank):
    table=evaluation_table(bank);fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[1.5,1]})
    width=.26;x=np.arange(len(FAMILIES))
    for k,(mode,colour) in enumerate((('policy',POLICY),('expert',EXPERT),('random',RANDOM))):
        values=[np.mean([row.get(mode,0) for row in table[f].values()]) for f in FAMILIES]
        bars=ax.bar(x+(k-1)*width,values,width,color=colour,label=t['modes'][mode])
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.015,f'{v:.2f}',ha='center',fontsize=7.5)
    ax.set_xticks(x);ax.set_xticklabels([t['family'][f] for f in FAMILIES]);ax.set_ylim(0,1.1);ax.set_ylabel(t['success']);ax.legend(frameon=False,fontsize=8)
    ax.set_title(t['eval_title']);ax.grid(axis='x',alpha=0)
    report=json.loads((bank/'aerial'/'evaluation.json').read_text());variants=('rotating','static');x=np.arange(len(variants))
    for k,(mode,colour) in enumerate((('policy',POLICY),('expert',EXPERT),('random',RANDOM))):
        values=[report['modes'][mode]['by_variant'].get(v,0) for v in variants]
        bx.bar(x+(k-1)*width,values,width,color=colour,label=t['modes'][mode])
        median=report['modes'][mode]['median_first_hit_s']
        if median:bx.text(.97,.94-.08*k,t['first_hit'].format(v=median),color=colour,fontsize=8.5,ha='right',transform=bx.transAxes)
    bx.set_xticks(x);bx.set_xticklabels([t[v] for v in variants]);bx.set_ylabel(t['strikes']);bx.set_title(t['aerial_title']);bx.grid(axis='x',alpha=0)
    fig.tight_layout();fig.savefig(out/'eval-summary.png',bbox_inches='tight');plt.close(fig)


def fig_throughput(t,out):
    # Measured on this machine during the 2026-10-19 runs (see docs/multi-robot.md).
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12.5,4.3),gridspec_kw={'width_ratios':[1,1.35]})
    ax.plot([1,4,8,10],[194,342,401,367],'o-',color=EXPERT,lw=2,label=t['sync']);ax.plot([6,8,10,12],[754,767,739,682],'o-',color=POLICY,lw=2,label=t['async'])
    ax.set_xlabel(t['workers']);ax.set_ylabel(t['sps']);ax.legend(frameon=False,fontsize=8);ax.set_ylim(0,850);ax.set_title(t['family']['navigation'])
    before=[13.2,22.,70.,40.];after=[3.5,.01,9.,135.];y=np.arange(len(before))
    bx.barh(y+.2,before,.38,color='#c0392b',label=t['before']);bx.barh(y-.2,after,.38,color=POLICY,label=t['after'])
    for k,(b,a) in enumerate(zip(before,after)):
        bx.text(b*1.15,k+.2,f'{b:g}',va='center',fontsize=8);bx.text(a*1.15,k-.2,f'{a:g}',va='center',fontsize=8)
    bx.set_xscale('log');bx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_:f'{v:g}'));bx.set_yticks(y);bx.set_yticklabels(t['bottlenecks'],fontsize=8);bx.invert_yaxis();bx.legend(frameon=False,fontsize=8,loc='upper right');bx.grid(axis='y',alpha=0)
    fig.suptitle(t['tp_title'],fontweight='bold');fig.tight_layout();fig.savefig(out/'throughput.png',bbox_inches='tight');plt.close(fig)


def fig_team(t,out):
    configs=('doctrine-expert','doctrine-rl','llm-rl');fig,(ax,bx)=plt.subplots(1,2,figsize=(12.5,4.4));ticks=[];tallest=0
    for c,config in enumerate(configs):
        report=json.loads((MATCHES/config/'report.json').read_text());ticks.append(c*4+(len(report['matches'])-1)/2)
        for k,match in enumerate(report['matches']):
            pos=c*4+k
            ax.bar(pos-.2,match['score']['red']['control'],.4,color=RED);ax.bar(pos+.2,match['score']['blue']['control'],.4,color=BLUE)
            bx.bar(pos-.2,match['score']['red']['attack_damage'],.4,color=RED);bx.bar(pos+.2,match['score']['blue']['attack_damage'],.4,color=BLUE)
            top=max(match['score']['red']['control'],match['score']['blue']['control']);tallest=max(tallest,top)
            ax.text(pos,top+6,t['win'] if match['winner']=='red' else t['loss'],ha='center',fontsize=8,color=RED if match['winner']=='red' else BLUE,fontweight='bold')
    ax.set_ylim(0,tallest*1.12)
    for axis in (ax,bx):axis.set_xticks(ticks);axis.set_xticklabels(t['configs'],fontsize=8.5);axis.grid(axis='x',alpha=0)
    ax.set_ylabel(t['control']);bx.set_ylabel(t['damage']);ax.set_title(t['control']);bx.set_title(t['damage'])
    ax.legend([Rectangle((0,0),1,1,fc=RED),Rectangle((0,0),1,1,fc=BLUE)],[t['winner']['red'],t['winner']['blue']],frameon=False,fontsize=8)
    fig.suptitle(t['team_title'],fontweight='bold');fig.tight_layout();fig.savefig(out/'team-matches.png',bbox_inches='tight');plt.close(fig)


def fig_305(t,out):
    base=ROOT/'checkpoints'/'serial305';items=[]
    for key,path in (('combat','combat/evaluation.json'),('defense','defense/evaluation.json'),('support','support/evaluation.json'),('mobility','mobility/evaluation.json'),
                     ('flight','flight/evaluation.json'),('flight_heldout','flight/evaluation-heldout.json'),('navigation_heldout','navigation/evaluation-heldout.json'),('navigation_composed','navigation/evaluation-composed.json')):
        report=json.loads((base/path).read_text());rate=report.get('success_rate')
        if rate is None:rate=report['successes']/report['episodes']
        items.append((t['s305'][key],rate,report.get('episodes')))
    fig,ax=plt.subplots(figsize=(9,3.8));y=np.arange(len(items))
    bars=ax.barh(y,[v for _,v,_ in items],color=POLICY);ax.set_yticks(y);ax.set_yticklabels([n for n,_,_ in items]);ax.invert_yaxis();ax.set_xlim(0,1.08)
    for bar,(_,v,n) in zip(bars,items):ax.text(v+.01,bar.get_y()+bar.get_height()/2,f'{v:.2f}  (n={n})',va='center',fontsize=8)
    ax.set_xlabel(t['success']);ax.set_title(t['s305_title']);ax.grid(axis='y',alpha=0);fig.tight_layout();fig.savefig(out/'serial305-evaluation.png',bbox_inches='tight');plt.close(fig)


def svg_architecture(t,out):
    a=t['arch'];font="font-family=\"PingFang SC,Hiragino Sans GB,Microsoft YaHei,Noto Sans CJK SC,Helvetica,Arial,sans-serif\""
    def box(x,y,w,h,fill,title,sub='',sub2='',stroke='#333'):
        parts=[f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
               f'<text x="{x+w/2}" y="{y+24}" text-anchor="middle" font-size="16" font-weight="700" {font}>{title}</text>']
        if sub:parts.append(f'<text x="{x+w/2}" y="{y+46}" text-anchor="middle" font-size="12" fill="#333" {font}>{sub}</text>')
        if sub2:parts.append(f'<text x="{x+w/2}" y="{y+64}" text-anchor="middle" font-size="12" fill="#333" {font}>{sub2}</text>')
        return ''.join(parts)
    def arrow(x1,y1,x2,y2,label='',dx=8):
        mid=((x1+x2)/2+dx,(y1+y2)/2)
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#444" stroke-width="1.8" marker-end="url(#a)"/>'+
                (f'<text x="{mid[0]}" y="{mid[1]+4}" font-size="11.5" fill="#555" {font}>{label}</text>' if label else ''))
    W=980;H=640
    body=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
          '<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#444"/></marker></defs>',
          f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
          box(240,18,500,64,'#fdebd0',a[0],a[1]),arrow(490,82,490,118,a[13]),
          box(170,120,640,92,'#fef9e7',a[2],a[3],a[4]),arrow(490,212,490,248,a[14]),
          box(120,250,740,64,'#e8f6f3',a[5],a[6]+'  ·  '+a[7]),
          arrow(330,314,330,352),arrow(690,314,690,352),
          box(120,354,420,74,'#e9f7ef',a[8],a[9]),box(580,354,280,74,'#e9f7ef',a[10],a[20]),
          arrow(330,428,330,468,a[15]),arrow(720,428,720,468,a[15]),
          box(120,470,740,64,'#eaf2f8',a[11],a[12]),
          f'<path d="M 860 502 C 945 502 945 282 860 282" fill="none" stroke="#888" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#a)"/>',
          f'<text x="940" y="395" font-size="11.5" fill="#666" text-anchor="end" transform="rotate(90 940 395)" {font}>{a[16]}</text>',
          box(120,560,740,64,'#f4ecf7',a[17],a[18],stroke='#7d3c98'),
          f'<path d="M 120 592 L 70 592 L 70 391 L 116 391" fill="none" stroke="#7d3c98" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#a)"/>',
          f'<text x="58" y="492" font-size="11.5" fill="#7d3c98" text-anchor="middle" transform="rotate(-90 58 492)" {font}>{a[19]}</text>',
          '</svg>']
    (out/'architecture.svg').write_text(''.join(body),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lang',nargs='+',default=['zh','en']);parser.add_argument('--out',default=str(ROOT/'docs'/'images'))
    args=parser.parse_args();bank=ROOT/'checkpoints'/'ground';geometry=arena()
    trace_path=MATCHES/'traces'/'llm-rl-9300001.json.gz';trace=load_json(trace_path)
    decisions=[json.loads(line) for line in (MATCHES/'llm-rl'/'decisions-9300001.jsonl').read_text().splitlines() if line.strip()]
    winner=json.loads((MATCHES/'llm-rl'/'report.json').read_text())['matches'][1]['winner']
    for lang in args.lang:
        out=Path(args.out)/lang;out.mkdir(parents=True,exist_ok=True);setup(lang);t=T[lang]
        svg_architecture(t,out);fig_field(t,out,geometry);fig_trajectories(t,out,geometry,trace,9300001,winner);fig_timeline(t,out,decisions,trace,9300001)
        fig_curves(t,out,bank);fig_heatmap(t,out,bank);fig_eval_summary(t,out,bank);fig_throughput(t,out);fig_team(t,out);fig_305(t,out)
        print('wrote',sorted(p.name for p in out.iterdir()))


if __name__=='__main__':main()
