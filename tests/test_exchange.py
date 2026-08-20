import unittest
from market.exchange import Exchange
from market.orders import Order


class TestExchange(unittest.TestCase):
    def test_partial_fill_execution_price_and_parties(self):
        x=Exchange(); x.submit(Order(1,"seller","SELL",2,0,101)); incoming=Order(2,"buyer","BUY",1,1,102); x.submit(incoming)
        t=x.trades[0]; self.assertEqual(t["price"],101); self.assertEqual(t["buyer_agent"],"buyer"); self.assertEqual(t["seller_agent"],"seller")
        self.assertEqual(t["aggressor_side"],"BUY"); self.assertAlmostEqual(x.book.depth("SELL"),1)

    def test_market_full_fill_and_fifo(self):
        x=Exchange(); x.submit(Order(1,"a","SELL",1,0,101)); x.submit(Order(2,"b","SELL",1,1,101)); x.submit(Order(3,"c","BUY",2,2))
        self.assertEqual([t["seller_agent"] for t in x.trades],["a","b"])


if __name__=="__main__": unittest.main()

