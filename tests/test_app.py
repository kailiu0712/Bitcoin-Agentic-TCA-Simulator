import unittest
from app.main import ALGORITHMS,REFERENCE_ALGORITHM,SimulationRequest,load_stylized_facts,run_execution_simulation

KEYS={item["key"] for item in ALGORITHMS}


class TestApplication(unittest.TestCase):
    def test_execution_api_service_measures_every_registered_algorithm(self):
        request=SimulationRequest(side="SELL",parent_quantity=.001,duration_seconds=30,
            volatility=2.02,spread=.1,arrival_rate=.536,displayed_depth=.3,seed=44)
        updates=[];result=run_execution_simulation(request,lambda p,m:updates.append((p,m)))
        self.assertGreater(len(result["execution_path"]),0);self.assertAlmostEqual(result["summary"]["completion"],1.0,places=8)
        self.assertEqual(set(result["comparison"]),KEYS)
        self.assertEqual(set(result["execution_paths"]),KEYS)
        self.assertEqual(result["summary"]["strategy"],REFERENCE_ALGORITHM)
        self.assertIn("adaptive",result["comparison"])
        self.assertTrue(all(result["execution_path"][i]["elapsed_seconds"]<=result["execution_path"][i+1]["elapsed_seconds"] for i in range(len(result["execution_path"])-1)))
        self.assertGreaterEqual(max(p for p,_ in updates),96)

    def test_request_validation_rejects_invalid_quantity(self):
        with self.assertRaises(ValueError):SimulationRequest(side="SELL",parent_quantity=0,duration_seconds=30,volatility=2,spread=.1,arrival_rate=.5,displayed_depth=.3,seed=1)

    def test_large_sell_case_handles_missing_midpoints(self):
        request=SimulationRequest(side="SELL",parent_quantity=5.0,duration_seconds=900,
            volatility=2.6,spread=.14,arrival_rate=.7,displayed_depth=.42,seed=2028)
        result=run_execution_simulation(request)
        self.assertEqual(set(result["comparison"]),KEYS)
        self.assertGreater(len(result["execution_path"]),0)

    def test_immediate_execution_moves_price_further_than_a_schedule(self):
        request=SimulationRequest(side="BUY",parent_quantity=4.5,duration_seconds=600,
            volatility=2.4,spread=.13,arrival_rate=.58,displayed_depth=.18,seed=2027)
        result=run_execution_simulation(request)
        immediate=result["comparison"]["immediate"];twap=result["comparison"]["twap"]
        self.assertGreater(abs(immediate["impact_bp"]),abs(twap["impact_bp"]))
        self.assertLess(twap["penalized_cost_bp"],immediate["penalized_cost_bp"])

    def test_impact_summary_reports_peak_terminal_and_post_trade(self):
        request=SimulationRequest(side="SELL",parent_quantity=.5,duration_seconds=120,
            volatility=2.02,spread=.1,arrival_rate=.536,displayed_depth=.3,seed=7)
        entry=run_execution_simulation(request)["comparison"]["twap"]
        for field in ("impact_bp","terminal_impact_bp","post_trade_impact_bp"):
            self.assertIsInstance(entry[field],float)
        self.assertGreaterEqual(abs(entry["impact_bp"]),abs(entry["terminal_impact_bp"])-1e-9)

    def test_negligible_parent_causes_no_measurable_impact(self):
        """A parent too small to move any price must measure as zero impact.

        This is the guard against paired-grid sampling artifacts: the control and
        execution markets emit events at different instants, so any interpolation
        that is not a zero-order hold leaks background price moves into the
        difference and reports them as execution impact.
        """
        base=dict(side="BUY",duration_seconds=600,volatility=2.4,spread=.13,
            arrival_rate=.58,displayed_depth=.18,seed=2027)
        tiny=run_execution_simulation(SimulationRequest(**base,parent_quantity=1e-5))["comparison"]["twap"]
        self.assertAlmostEqual(tiny["impact_bp"],0.0,places=9)
        self.assertAlmostEqual(tiny["terminal_impact_bp"],0.0,places=9)
        real=run_execution_simulation(SimulationRequest(**base,parent_quantity=4.5))["comparison"]["twap"]
        self.assertGreater(abs(real["impact_bp"]),abs(tiny["impact_bp"]))

    def test_urgent_execution_peaks_above_schedule_on_every_preset(self):
        presets=[dict(side="SELL",parent_quantity=1.,duration_seconds=300,volatility=2.02,
                      spread=.10,arrival_rate=.536,displayed_depth=.30,seed=2026),
                 dict(side="BUY",parent_quantity=4.5,duration_seconds=600,volatility=2.40,
                      spread=.13,arrival_rate=.580,displayed_depth=.18,seed=2027)]
        for preset in presets:
            comparison=run_execution_simulation(SimulationRequest(**preset))["comparison"]
            self.assertGreater(abs(comparison["immediate"]["impact_bp"]),
                               abs(comparison["twap"]["impact_bp"]),preset)

    def test_stylized_fact_evidence_is_complete(self):
        evidence=load_stylized_facts()
        head=evidence["headline"]
        self.assertEqual(head["facts"],len(evidence["facts"]))
        self.assertEqual(head["gate_metrics"],len(evidence["gate"]))
        self.assertLessEqual(head["worst_distance"],head["tolerance"])
        for fact in evidence["facts"]:
            for field in ("id","name","name_en","why","value","target","detail"):
                self.assertTrue(fact[field],f"{fact['id']} missing {field}")
        self.assertAlmostEqual(evidence["impact_law"]["delta"],0.5,delta=0.05)
        self.assertGreater(len(evidence["impact_law"]["curve"]),4)

    def test_architecture_layers_declare_mechanisms_and_facts(self):
        layers=load_stylized_facts()["architecture"]
        self.assertEqual(len(layers),3)
        for layer in layers:
            self.assertTrue(layer["mechanisms"] and layer["facts"])


if __name__=="__main__":unittest.main()
