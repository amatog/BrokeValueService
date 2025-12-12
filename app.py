import os

from flask import Flask, jsonify, request

from data_layer import DataLayer
from strategies import (
    graham_valuation,
    buffett_quality,
    greenblatt_magic_formula,
    combined_value_score,
    munger_quality,
    lynch_growth_value,
    schloss_deep_value,
    davis_growth_quality,
    templeton_contrarian_value,
    klarman_margin_of_safety,
)

app = Flask(__name__)
data_layer = DataLayer()


@app.route("/", methods=["GET"])
def root():
    # Lokal:  http://127.0.0.1:8001/
    # Server: https://value.netdesign.ch/api/
    return "Hello from value-services root"


@app.route("/health", methods=["GET"])
def health():
    # Lokal:  http://127.0.0.1:8001/health
    # Server: https://value.netdesign.ch/api/health
    return jsonify(status="ok", service="value-services")


def _require_symbol():
    symbol = request.args.get("symbol")
    if not symbol:
        return None, jsonify(error="Missing required query parameter: symbol"), 400
    return symbol.upper(), None, None


@app.route("/debug/env", methods=["GET"])
def debug_env():
    alpha_keys = [k for k in os.environ.keys() if "ALPHA" in k]
    return jsonify(
        has_key="ALPHAVANTAGE_API_KEY" in os.environ,
        value=os.environ.get("ALPHAVANTAGE_API_KEY"),
        keys=alpha_keys,
    )


@app.route("/graham", methods=["GET"])
def value_graham():
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    g = graham_valuation(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, graham=g)


@app.route("/buffett", methods=["GET"])
def value_buffett():
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    b = buffett_quality(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, buffett=b)


@app.route("/greenblatt", methods=["GET"])
def value_greenblatt():
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    gr = greenblatt_magic_formula(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, greenblatt=gr)

@app.route("/munger", methods=["GET"])
def value_munger():
    # z.B. http://127.0.0.1:8001/munger?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    m = munger_quality(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, munger=m)

@app.route("/lynch", methods=["GET"])
def value_lynch():
    # z.B. http://127.0.0.1:8001/lynch?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    l = lynch_growth_value(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, lynch=l)


@app.route("/schloss", methods=["GET"])
def value_schloss():
    # z.B. http://127.0.0.1:8001/schloss?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    s = schloss_deep_value(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, schloss=s)


@app.route("/davis", methods=["GET"])
def value_davis():
    # z.B. http://127.0.0.1:8001/davis?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    d = davis_growth_quality(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, davis=d)


@app.route("/templeton", methods=["GET"])
def value_templeton():
    # z.B. http://127.0.0.1:8001/templeton?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    t = templeton_contrarian_value(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, templeton=t)


@app.route("/klarman", methods=["GET"])
def value_klarman():
    # z.B. http://127.0.0.1:8001/klarman?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    k = klarman_margin_of_safety(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, klarman=k)



@app.route("/score", methods=["GET"])
def value_score():
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)

    g = graham_valuation(fundamentals)
    b = buffett_quality(fundamentals)
    gr = greenblatt_magic_formula(fundamentals)
    m = munger_quality(fundamentals)
    l = lynch_growth_value(fundamentals)
    s = schloss_deep_value(fundamentals)
    d = davis_growth_quality(fundamentals)
    t = templeton_contrarian_value(fundamentals)
    k = klarman_margin_of_safety(fundamentals)
    score = combined_value_score(g, b, gr, fundamentals)

    return jsonify(
        symbol=symbol,
        fundamentals=fundamentals,
        graham=g,
        buffett=b,
        greenblatt=gr,
        munger=m,
        lynch=l,
        schloss=s,
        davis=s,
        templeton=t,
        klarman=k,
        score=score,
    )






