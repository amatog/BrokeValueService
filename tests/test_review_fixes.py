import os
import unittest
from unittest.mock import patch

from data_layer import DataLayer
from app import app
from backtest.engines import DirectStrategyEngine, ValueRestEngine


class ReviewFixTests(unittest.TestCase):
    def test_secret_is_not_disclosed(self):
        with patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "secret-test-value"}):
            response = app.test_client().get("/debug/env")
        self.assertNotIn(b"secret-test-value", response.data)
        self.assertNotIn("value", response.json)

    def test_statement_failure_does_not_retry_or_fabricate(self):
        layer = DataLayer()
        layer.use_dummy = False
        for method in (layer.get_income, layer.get_balance, layer.get_cashflow):
            with patch.object(layer, "_get", side_effect=RuntimeError("offline")) as get:
                self.assertEqual(method("AAPL"), {"annualReports": [], "source": "unavailable"})
                get.assert_called_once()

    def test_limit_response_is_rejected(self):
        layer = DataLayer()
        layer.api_key = "test"
        with patch("data_layer.requests.get") as get:
            get.return_value.json.return_value = {"Information": "rate limit"}
            with self.assertRaises(ValueError):
                layer._get({"symbol": "AAPL"})

    def test_historical_data_is_rejected_before_fetch(self):
        for engine in (DirectStrategyEngine(DataLayer()), ValueRestEngine("http://invalid")):
            with self.assertRaises(ValueError):
                engine.snapshot("AAPL", "2020-01-01")
        self.assertEqual(app.test_client().get("/bundle?symbol=AAPL&asof=2020-01-01").status_code, 422)

    def test_lynch_endpoints_with_offline_data(self):
        layer = DataLayer()
        layer.use_dummy = True
        with patch("app.data_layer", layer):
            client = app.test_client()
            self.assertEqual(client.get("/lynch?symbol=AAPL").status_code, 200)
            self.assertEqual(client.get("/debug/keys?symbol=AAPL").status_code, 200)


if __name__ == "__main__":
    unittest.main()
