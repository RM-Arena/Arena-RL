"""Replace request_llm() with a model call; execution continues during that call.

Run from Arena-RL: uv run python examples/tactical_client.py --policies checkpoints/tactics
This example issues scripted decisions. It does not call an LLM or claim LLM performance.
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time


def request_llm(scene):
    # Emulate a slow inference call. The 10 Hz controller keeps its previous command.
    time.sleep(1.5)
    heavy=next((e for e in scene['visible_enemies'] if e.get('role') in ('heavy','hero') and e['alive']),None)
    if heavy:
        return {'id':f'pursue-{scene["tick"]}','skill':'pursue_target','target_id':heavy['id'],
                'preferred_range_m':2.5,'leash_radius_m':8.,'timeout_s':15.,
                'basis_tick':scene['tick'],'max_staleness_s':3.}
    return {'id':f'move-{scene["tick"]}','skill':'navigate','point':[4.5,5.15],
            'timeout_s':30.,'basis_tick':scene['tick'],'max_staleness_s':3.}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policies',required=True);args=parser.parse_args()
    process=subprocess.Popen([sys.executable,'-m','arena_rl','serve-tactics','--policies',args.policies,'--realtime'],
                             stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    messages=queue.Queue()
    def receive():
        for line in process.stdout:messages.put(json.loads(line))
    threading.Thread(target=receive,daemon=True).start()
    def send(value):process.stdin.write(json.dumps(value)+'\n');process.stdin.flush()
    try:
        ready=messages.get(timeout=30);print('Ready:',ready['protocol'])
        scene=ready['states'][0]
        for n in range(3):
            command=request_llm(scene)
            send({'id':n,'op':'command','command':command})
            while True:
                message=messages.get(timeout=30)
                print(message.get('type'),message.get('id'))
                if message.get('type')=='state':scene=message['states'][0];break
        send({'op':'close'});process.wait(timeout=20)
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=20)
        process.stdin.close();process.stdout.close()


if __name__=='__main__':main()
