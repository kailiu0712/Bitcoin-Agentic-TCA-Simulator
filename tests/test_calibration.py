import json
import unittest
from pathlib import Path
import pandas as pd
import yaml
from analytics.stylized_facts import compute
from calibration.calibrate import load_calibration_target
from calibration.objective import _aligned_curve_nrmse,_nrmse,calibration_loss
from scripts.run_simulation import simulate


class TestCalibration(unittest.TestCase):
    def test_distribution_and_curve_distance(self):
        self.assertEqual(_nrmse([1,2,3],[1,2,3]),0)
        self.assertGreater(_nrmse([2,3,4],[1,2,3]),0)
        sim=[{"flow":1,"change":1},{"flow":2,"change":2},{"flow":3,"change":3}]
        target=[{"flow":1.5,"change":1.5},{"flow":2.5,"change":2.5},{"flow":3.5,"change":3.5}]
        self.assertAlmostEqual(_aligned_curve_nrmse(sim,target,"flow","change"),0.0)

    def test_deterministic_objective(self):
        cfg=yaml.safe_load(Path("config/default.yaml").read_text());simcfg=cfg["simulation"].copy();simcfg["duration_seconds"]=30
        target=compute(simulate(simcfg,123)[0]);a=calibration_loss(compute(simulate(simcfg,42)[0]),target,cfg["calibration"]["weights"])[0];b=calibration_loss(compute(simulate(simcfg,42)[0]),target,cfg["calibration"]["weights"])[0]
        self.assertEqual(a,b)

    def test_calibration_target_has_no_validation_branch(self):
        target=load_calibration_target();self.assertIn("n_trades",target);self.assertNotIn("validation",target)
        config=yaml.safe_load(Path("config/default.yaml").read_text())["data"]
        self.assertTrue(set(config["calibration_days"]).isdisjoint(config["validation_days"]))


if __name__=="__main__":unittest.main()
