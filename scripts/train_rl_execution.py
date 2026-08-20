import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,yaml
from liquidation.runner import make_agent,run_market,midpoint_at,midpoint_from_path,midpoint_path


def key_string(key):return "|".join(map(str,key))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="outputs/calibration/best_config.yaml");ap.add_argument("--episodes",type=int,default=600);ap.add_argument("--output-dir",default="outputs/liquidation/rl2");ap.add_argument("--strategy",choices=["rl","rl2"],default="rl2");ap.add_argument("--alpha",type=float,default=.10);ap.add_argument("--gamma",type=float,default=.97);ap.add_argument("--terminal-penalty",type=float,default=.05)
    args=ap.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())["simulation"];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(91);q={};history=[];started=time.perf_counter()
    alpha,gamma=args.alpha,args.gamma;start=180.;post=60.
    for ep in range(args.episodes):
        duration=float(rng.choice([120,300,600]));side=str(rng.choice(["BUY","SELL"]));ratio=float(np.exp(rng.uniform(np.log(.001),np.log(.2))))
        baseline,_=run_market(cfg,50_000+ep,start+duration+post);base_path=midpoint_path(baseline.events);volume=sum(t["size"] for t in baseline.trades if start<=t["timestamp"]<=start+duration);quantity=max(1e-5,ratio*volume)
        agent=make_agent(args.strategy,side,quantity,start,start+duration,max(4,int(duration/10)),q_table=q,seed=ep);agent.epsilon=max(.03,.40*(1-ep/args.episodes))
        ex,agent=run_market(cfg,50_000+ep,start+duration+post,agent);arrival=agent.arrival_mid or midpoint_at(ex.events,start)
        transitions=[]
        for i,((state,action),rec) in enumerate(zip(agent.actions,agent.records)):
            baseline_mid=midpoint_from_path(base_path,rec["timestamp"]);fill_price=rec["fill_notional"]/rec["filled"] if rec["filled"] else baseline_mid
            reward=-agent.sign*(fill_price-baseline_mid)/arrival*(rec["filled"]/quantity)
            if i==len(agent.records)-1:reward-=args.terminal_penalty*(agent.remaining/quantity)**2
            next_state=agent.actions[i+1][0] if i+1<len(agent.actions) else None;action_count=len(agent.ACTIONS);values=q.setdefault(state,[0.0]*action_count);future=0 if next_state is None else max(q.setdefault(next_state,[0.0]*action_count));values[action]+=alpha*(reward+gamma*future-values[action]);transitions.append(reward)
        history.append({"episode":ep,"side":side,"duration":duration,"participation":ratio,"quantity":quantity,"completion":agent.filled/quantity,"reward":sum(transitions)})
    serial={key_string(k):v for k,v in q.items()};(out/"q_table.json").write_text(json.dumps(serial,indent=2));import pandas as pd;pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    (out/"training_summary.json").write_text(json.dumps({"strategy":args.strategy,"episodes":args.episodes,"states":len(q),"actions":agent.ACTIONS,"alpha":alpha,"gamma":gamma,"terminal_penalty":args.terminal_penalty,"reward":"negative paired execution cost plus terminal inventory penalty","runtime_seconds":time.perf_counter()-started,"seed":91},indent=2));print(out/"q_table.json")


if __name__=="__main__":main()
