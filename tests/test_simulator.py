import unittest
import yaml
from pathlib import Path
from scripts.run_simulation import simulate


class TestSimulator(unittest.TestCase):
    def test_deterministic_and_valid(self):
        cfg=yaml.safe_load(Path("config/default.yaml").read_text())["simulation"]; cfg["duration_seconds"]=10
        a,_=simulate(cfg,42); b,_=simulate(cfg,42)
        self.assertTrue(a.equals(b)); self.assertTrue(a.timestamp_ns.is_monotonic_increasing)
        q=a.dropna(subset=["best_bid","best_ask"]); self.assertTrue((q.best_bid<q.best_ask).all())


if __name__=="__main__": unittest.main()

