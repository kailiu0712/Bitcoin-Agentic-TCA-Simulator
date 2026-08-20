import unittest
import pandas as pd
from analytics.stylized_facts import compute


class TestAnalytics(unittest.TestCase):
    def test_known_spread_sign_acf_response(self):
        rows=[]
        for i,(side,mid) in enumerate(zip(["BUY","BUY","SELL","SELL"],[100,101,102,101])):
            rows.append({"timestamp_ns":i*1_000_000_000,"event_type":"QUOTE_UPDATE","side":None,"price":None,"size":None,"best_bid":mid-1,"best_ask":mid+1,"bid_size":1,"ask_size":1,"mid":mid,"spread":2})
            rows.append({"timestamp_ns":i*1_000_000_000+1,"event_type":"TRADE","side":side,"price":mid,"size":1,"best_bid":mid-1,"best_ask":mid+1,"bid_size":1,"ask_size":1,"mid":mid,"spread":2})
        m=compute(pd.DataFrame(rows),lags=[1]); self.assertEqual(m["spread"]["median"],2); self.assertAlmostEqual(m["sign_acf"]["1"],1/3)
        self.assertAlmostEqual(m["response"]["1"],1.0)
        self.assertIn("clock_response",m)
        self.assertIn("relative_spread",m["distributions"])


if __name__=="__main__": unittest.main()
