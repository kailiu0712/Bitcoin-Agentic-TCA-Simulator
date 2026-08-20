import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from liquidation.agents import EnhancedRLExecutionAgent
from scripts.run_liquidation_experiments import load_q


def main():
    ap=argparse.ArgumentParser(description="Generate one market-order instruction from the tuned execution policy")
    ap.add_argument("--q-table",default="outputs/execution_algorithms/rl2_tuned/q_table.json")
    ap.add_argument("--side",choices=["BUY","SELL"],required=True);ap.add_argument("--parent-quantity",type=float,required=True);ap.add_argument("--remaining-quantity",type=float,required=True)
    ap.add_argument("--elapsed-seconds",type=float,required=True);ap.add_argument("--duration-seconds",type=float,required=True);ap.add_argument("--steps-remaining",type=int,required=True)
    ap.add_argument("--spread",type=float,required=True);ap.add_argument("--bid-depth",type=float,required=True);ap.add_argument("--ask-depth",type=float,required=True)
    ap.add_argument("--arrival-mid",type=float,required=True);ap.add_argument("--current-mid",type=float,required=True);args=ap.parse_args()
    if args.parent_quantity<=0 or not 0<args.remaining_quantity<=args.parent_quantity:raise ValueError("quantities must satisfy 0 < remaining <= parent")
    if args.duration_seconds<=0 or not 0<=args.elapsed_seconds<=args.duration_seconds:raise ValueError("elapsed time must be inside the execution window")
    sign=1 if args.side=="BUY" else -1;schedule=args.remaining_quantity/max(1,args.steps_remaining);depth=args.ask_depth if args.side=="BUY" else args.bid_depth
    obs={"time_fraction":args.elapsed_seconds/args.duration_seconds,"inventory_fraction":args.remaining_quantity/args.parent_quantity,"spread":args.spread,
        "side_adjusted_imbalance":sign*(args.bid_depth-args.ask_depth)/(args.bid_depth+args.ask_depth+1e-9),"opposite_depth_ratio":depth/(schedule+1e-12),
        "side_adjusted_momentum":sign*(args.current_mid-args.arrival_mid)/args.arrival_mid}
    instruction=EnhancedRLExecutionAgent.instruction_from_observation(obs,args.remaining_quantity,args.steps_remaining,load_q(args.q_table),args.steps_remaining==1)
    result={"order_type":"MARKET","side":args.side,"quantity":instruction["quantity"],"schedule_quantity":schedule,"multiplier":instruction["multiplier"],
        "state":list(instruction["state"]),"inputs":instruction["features"],"remaining_after_requested_fill":args.remaining_quantity-instruction["quantity"]}
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
