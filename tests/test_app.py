import unittest
from app.main import SimulationRequest,run_execution_simulation


class TestApplication(unittest.TestCase):
    def test_execution_api_service_returns_plot_path(self):
        request=SimulationRequest(side="SELL",parent_quantity=.001,duration_seconds=30,
            volatility=2.02,spread=.1,arrival_rate=.536,displayed_depth=.3,seed=44)
        updates=[];result=run_execution_simulation(request,lambda p,m:updates.append((p,m)))
        self.assertGreater(len(result["execution_path"]),0);self.assertAlmostEqual(result["summary"]["completion"],1.0,places=8)
        self.assertEqual(set(result["comparison"]),{"immediate","twap","adaptive"})
        self.assertEqual(result["summary"]["strategy"],"adaptive")
        self.assertTrue(all(result["execution_path"][i]["elapsed_seconds"]<=result["execution_path"][i+1]["elapsed_seconds"] for i in range(len(result["execution_path"])-1)))
        self.assertGreaterEqual(max(p for p,_ in updates),96)

    def test_request_validation_rejects_invalid_quantity(self):
        with self.assertRaises(ValueError):SimulationRequest(side="SELL",parent_quantity=0,duration_seconds=30,volatility=2,spread=.1,arrival_rate=.5,displayed_depth=.3,seed=1)


if __name__=="__main__":unittest.main()
