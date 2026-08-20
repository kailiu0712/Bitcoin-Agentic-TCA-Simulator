import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,optuna,pandas as pd,yaml
from analytics.stylized_facts import compute
from calibration.objective import calibration_loss
from scripts.run_simulation import simulate


def load_calibration_target(path="outputs/baseline_market/real_stylized_facts/pooled_targets.json"):
    return json.loads(Path(path).read_text())["calibration"]


def suggest(trial,base):
    x=dict(base)
    x.update(market_order_arrival_rate=trial.suggest_float("market_order_arrival_rate",.25,.75),
        trade_sign_persistence=trial.suggest_float("trade_sign_persistence",.72,.95),
        sign_regime_switch_probability=trial.suggest_float("sign_regime_switch_probability",.01,.16),
        trade_size_median=trial.suggest_float("trade_size_median",.0004,.003,log=True),
        trade_size_sigma=trial.suggest_float("trade_size_sigma",2.0,3.5),
        market_maker_quote_size=trial.suggest_float("market_maker_quote_size",.08,1.5,log=True),
        market_maker_quote_size_sigma=trial.suggest_float("market_maker_quote_size_sigma",.2,1.8),
        market_maker_level_spacing_ticks=trial.suggest_int("market_maker_level_spacing_ticks",1,5),
        market_maker_refresh_seconds=trial.suggest_float("market_maker_refresh_seconds",.1,.8,log=True),
        market_maker_inventory_skew=trial.suggest_float("market_maker_inventory_skew",.01,2.0,log=True),
        market_maker_inventory_signal_decay=trial.suggest_float("market_maker_inventory_signal_decay",.05,3.0,log=True),
        market_maker_flow_response=trial.suggest_float("market_maker_flow_response",.05,10.0,log=True),
        market_maker_flow_signal_decay=trial.suggest_float("market_maker_flow_signal_decay",.05,5.0,log=True),
        market_maker_spread_sensitivity=trial.suggest_float("market_maker_spread_sensitivity",.001,3.0,log=True),
        market_maker_asymmetric_liquidity=trial.suggest_float("market_maker_asymmetric_liquidity",.0,1.5),
        reference_price_volatility=trial.suggest_float("reference_price_volatility",2.0,25.0,log=True),
        order_flow_burst_probability=trial.suggest_float("order_flow_burst_probability",.05,.7),
        order_flow_burst_interval_seconds=trial.suggest_float("order_flow_burst_interval_seconds",.001,.1,log=True))
    return x


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="config/default.yaml");ap.add_argument("--targets",default="outputs/baseline_market/real_stylized_facts/pooled_targets.json");ap.add_argument("--output-dir",default="outputs/calibration");ap.add_argument("--trials",type=int)
    args=ap.parse_args();full=yaml.safe_load(Path(args.config).read_text());base=full["simulation"];ccfg=full["calibration"];target=load_calibration_target(args.targets);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();sampler=optuna.samplers.TPESampler(seed=base["seed"],n_startup_trials=15,multivariate=True);study=optuna.create_study(direction="minimize",sampler=sampler)
    initial_cfg=dict(base)
    def objective(trial):
        cfg=initial_cfg if trial.number==0 else suggest(trial,base);losses=[];all_rows=[]
        for seed_index in range(ccfg["seeds_per_trial"]):
            df,meta=simulate(cfg,1_000_000+trial.number*100+seed_index)
            if len(df)==0 or meta["number_of_trades"]<100:raise optuna.TrialPruned("insufficient trades")
            loss,rows=calibration_loss(compute(df),target,ccfg["weights"]);losses.append(loss);all_rows.append(rows)
        trial.set_user_attr("seed_losses",losses);trial.set_user_attr("loss_std",float(np.std(losses)))
        by_metric={}
        for rows in all_rows:
            for row in rows:by_metric.setdefault(row["metric"],[]).append(row["normalized_distance"])
        trial.set_user_attr("metric_distances",{k:float(np.mean(v)) for k,v in by_metric.items()})
        return float(np.mean(losses)+.05*np.std(losses))
    study.optimize(objective,n_trials=args.trials or ccfg["trials"],show_progress_bar=False,gc_after_trial=True)
    best_cfg=dict(base);best_cfg.update(study.best_trial.params)
    trials=study.trials_dataframe(attrs=("number","value","params","user_attrs","state"));trials.to_parquet(out/"trials.parquet",index=False);trials.to_csv(out/"trials.csv",index=False)
    (out/"best_config.yaml").write_text(yaml.safe_dump({"simulation":best_cfg,"frozen":False,"calibration_dates":full["data"]["calibration_days"]},sort_keys=False))
    errors=pd.DataFrame([{"metric":k,"normalized_distance":v} for k,v in study.best_trial.user_attrs["metric_distances"].items()]).sort_values("normalized_distance",ascending=False);errors.to_csv(out/"metric_errors.csv",index=False)
    plt.figure();plt.plot(trials.number,trials.value,alpha=.5);plt.plot(trials.number,trials.value.cummin(),label="best");plt.xlabel("Trial");plt.ylabel("Loss");plt.legend();plt.tight_layout();plt.savefig(out/"calibration_history.png",dpi=150);plt.close()
    summary={"method":"Optuna TPE","trials":len(study.trials),"seeds_per_trial":ccfg["seeds_per_trial"],"best_loss":study.best_value,"best_trial":study.best_trial.number,"best_loss_std":study.best_trial.user_attrs["loss_std"],"runtime_seconds":time.perf_counter()-started,"calibration_dates":full["data"]["calibration_days"],"validation_dates_not_loaded":full["data"]["validation_days"]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
