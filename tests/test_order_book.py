import unittest
from market.order_book import OrderBook
from market.orders import Order


class TestBook(unittest.TestCase):
    def test_priority_depth_cancel(self):
        b=OrderBook(); a=Order(1,"a","BUY",2,0,100); c=Order(2,"b","BUY",3,1,101); d=Order(3,"c","BUY",4,2,101)
        for x in (a,c,d): b.add(x)
        self.assertEqual(b.best_price("BUY"),101); self.assertEqual(b.best_order("BUY").order_id,2); self.assertEqual(b.depth("BUY"),7)
        self.assertTrue(b.cancel(2)); self.assertEqual(b.best_order("BUY").order_id,3)

    def test_invalid_quantity(self):
        with self.assertRaises(ValueError): Order(1,"a","BUY",0,0,100)


if __name__=="__main__": unittest.main()

