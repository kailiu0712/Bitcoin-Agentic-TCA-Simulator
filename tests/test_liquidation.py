import unittest
from pathlib import Path
import yaml
from liquidation.runner import make_agent,midpoint_at,run_market


class TestLiquidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=yaml.safe_load(Path("config/default.yaml").read_text())["simulation"].copy();cls.cfg["duration_seconds"]=20

    def test_twap_timing_direction_and_inventory(self):
        for side in ("BUY","SELL"):
            agent=make_agent("twap",side,.01,2,8,4);_,agent=run_market(self.cfg,12,10,agent)
            self.assertTrue(all(2<=r["timestamp"]<=8 for r in agent.records));self.assertTrue(all(a>=b for a,b in zip([.01]+[r["remaining"] for r in agent.records[:-1]],[r["remaining"] for r in agent.records])))
            self.assertGreater(agent.filled,0);self.assertTrue(all((t>=0) for t in [r["filled"] for r in agent.records]))

    def test_small_twap_completes(self):
        agent=make_agent("twap","SELL",.001,2,8,4);_,agent=run_market(self.cfg,13,10,agent);self.assertAlmostEqual(agent.filled,.001,places=8)

    def test_rl_action_bounds(self):
        agent=make_agent("rl","SELL",.01,2,8,4,q_table={});_,agent=run_market(self.cfg,14,10,agent);self.assertTrue(all(0<=a<3 for _,a in agent.actions))

    def test_enhanced_rl_state_actions_and_terminal_catchup(self):
        agent=make_agent("rl2","SELL",.01,2,8,4,q_table={});_,agent=run_market(self.cfg,14,10,agent)
        self.assertTrue(all(len(state)==6 and 0<=action<5 for state,action in agent.actions))
        self.assertEqual(agent.actions[0][1],2)
        self.assertAlmostEqual(agent.records[-1]["requested"],agent.records[-2]["remaining"],places=10)

    def test_additional_execution_agents_complete_small_order(self):
        for strategy in ("frontload","adaptive"):
            agent=make_agent(strategy,"BUY",.001,2,8,4);_,agent=run_market(self.cfg,31,10,agent)
            self.assertAlmostEqual(agent.filled,.001,places=8)

    def test_immediate_agent_submits_once_at_start(self):
        agent=make_agent("immediate","SELL",.001,2,8,4);_,agent=run_market(self.cfg,32,10,agent)
        self.assertEqual(len(agent.records),1);self.assertEqual(agent.records[0]["timestamp"],2);self.assertEqual(agent.records[0]["requested"],.001)

    def test_common_seed_pre_start_state(self):
        base,_=run_market(self.cfg,21,10);agent=make_agent("twap","BUY",.001,5,8,3);with_liq,_=run_market(self.cfg,21,10,agent)
        a=[(e["timestamp"],e["mid"]) for e in base.events if e["event_type"]=="QUOTE_UPDATE" and e["timestamp"]<5]
        b=[(e["timestamp"],e["mid"]) for e in with_liq.events if e["event_type"]=="QUOTE_UPDATE" and e["timestamp"]<5];self.assertEqual(a,b)

    def test_midpoint_lookup_uses_post_trade_book_state(self):
        events=[{"timestamp":1.0,"event_type":"QUOTE_UPDATE","mid":100.0},
                {"timestamp":2.0,"event_type":"MARKET_EXECUTION","mid":100.5}]
        self.assertEqual(midpoint_at(events,2.0),100.5)

    def test_midpoint_lookup_ignores_partial_requote_states(self):
        events=[{"timestamp":1.0,"event_type":"QUOTE_UPDATE","mid":100.0},
                {"timestamp":2.0,"event_type":"LIMIT_ADD","mid":105.0},
                {"timestamp":2.0,"event_type":"QUOTE_UPDATE","mid":101.0}]
        self.assertEqual(midpoint_at(events,2.0),101.0)


if __name__=="__main__":unittest.main()
