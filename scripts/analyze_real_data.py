import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd,yaml
from analytics.plots import make_plots
from analytics.scaling import fit_power_law
from analytics.stylized_facts import compute,scalar_metrics


def summarize_daily(daily,days):
    rows=[{"date":day,**scalar_metrics(daily[day])} for day in days]
    frame=pd.DataFrame(rows);summary={}
    for c in frame.columns[1:]:
        v=pd.to_numeric(frame[c],errors="coerce").dropna()
        summary[c]={"mean":float(v.mean()),"std":float(v.std(ddof=1)),"min":float(v.min()),"max":float(v.max()),"daily":dict(zip(frame.date,frame[c]))}
    return frame,summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="config/default.yaml");ap.add_argument("--output-dir",default="outputs/baseline_market/real_stylized_facts");ap.add_argument("--plots",action="store_true")
    args=ap.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());data_cfg=cfg["data"];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(data_cfg["processed_events"]);daily={}
    for day,g in df.groupby("date",sort=True):
        print("analyzing",day,flush=True);m=compute(g);m["scaling"]=fit_power_law(m["size_impact"]);m["clock_time_scaling"]={h:fit_power_law(p) for h,p in m["clock_time_size_impact"].items()};daily[day]=m
        if args.plots:make_plots(g,m,out/"plots",day)
    calibration_days=data_cfg["calibration_days"];validation_days=data_cfg["validation_days"]
    pooled={"calibration":compute(df[df.date.isin(calibration_days)]),"validation":compute(df[df.date.isin(validation_days)])}
    daily_table,daily_summary=summarize_daily(daily,calibration_days)
    _,validation_summary=summarize_daily(daily,validation_days)
    (out/"metrics_by_day.json").write_text(json.dumps(daily,indent=2));daily_table.to_csv(out/"metrics_by_day.csv",index=False)
    (out/"pooled_targets.json").write_text(json.dumps(pooled,indent=2));(out/"daily_variability.json").write_text(json.dumps({"calibration":daily_summary,"validation":validation_summary},indent=2))
    print(json.dumps({"calibration_days":calibration_days,"validation_days":validation_days,"daily_metrics":str(out/"metrics_by_day.csv")},indent=2))


if __name__=="__main__":main()
