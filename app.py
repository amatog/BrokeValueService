import os

from flask import Flask, jsonify, request

from data_layer import DataLayer
from strategies import (
    graham_strategy,
    buffett_strategy,
    greenblatt_strategy,
    munger_strategy,
    lynch_strategy,
    schloss_strategy,
    davis_strategy,
    templeton_strategy,
    klarman_strategy,
    combined_value_score,
    piotroski_f_score,
    beneish_penalty,
    montier_penalty,
    run_default_value_bundle,
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


@app.route("/debug/keys", methods=["GET"])
def debug_strategy_keys():
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)

    strategies = {
        "graham": graham_strategy(fundamentals),
        "buffett": buffett_strategy(fundamentals),
        "greenblatt": greenblatt_strategy(fundamentals),
        "munger": munger_strategy(fundamentals),
        "lynch": lynch_strategy(fundamentals),
        "schloss": schloss_strategy(fundamentals),
        "davis": davis_strategy(fundamentals),
        "templeton": templeton_strategy(fundamentals),
        "klarman": klarman_strategy(fundamentals),
    }

    return jsonify({
        name: sorted(list(result.keys()))
        for name, result in strategies.items()
    })


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
    g = graham_strategy(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, graham=g)


@app.route("/buffett", methods=["GET"])
def value_buffett():
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    b = buffett_strategy(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, buffett=b)


@app.route("/greenblatt", methods=["GET"])
def value_greenblatt():
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    gr = greenblatt_strategy(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, greenblatt=gr)

@app.route("/munger", methods=["GET"])
def value_munger():
    # z.B. http://127.0.0.1:8001/munger?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    m = munger_strategy(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, munger=m)

@app.route("/lynch", methods=["GET"])
def value_lynch():
    # z.B. http://127.0.0.1:8001/lynch?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    l = lynch_strategy(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, lynch=l)


@app.route("/schloss", methods=["GET"])
def value_schloss():
    # z.B. http://127.0.0.1:8001/schloss?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    s = schloss_strategy(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, schloss=s)


@app.route("/davis", methods=["GET"])
def value_davis():
    # z.B. http://127.0.0.1:8001/davis?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    d = davis_strategy(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, davis=d)


@app.route("/templeton", methods=["GET"])
def value_templeton():
    # z.B. http://127.0.0.1:8001/templeton?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    t = templeton_strategy(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, templeton=t)


@app.route("/klarman", methods=["GET"])
def value_klarman():
    # z.B. http://127.0.0.1:8001/klarman?symbol=AAPL
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    k = klarman_strategy(fundamentals)
    return jsonify(symbol=symbol, fundamentals=fundamentals, klarman=k)



@app.route("/score", methods=["GET"])
def value_score():
    symbol, error_response, status = _require_symbol()
    if error_response:
        return error_response, status

    # Optional: sector aus Query (für overrides / weights)
    sector = request.args.get("sector")

    # ----------------------------
    # Datenbeschaffung (Data Layer)
    # ----------------------------
    # Snapshot-Daten (für klassische Strategien)
    fundamentals = data_layer.get_basic_fundamentals(symbol)
    growth = data_layer.get_growth(symbol)

    # Perioden-Daten (für Piotroski/Beneish/Montier)
    income = data_layer.get_income(symbol)
    balance = data_layer.get_balance(symbol)
    cashflow = data_layer.get_cashflow(symbol)

    # ----------------------------
    # Einzelstrategien (Snapshot-based)
    # ----------------------------
    g = graham_strategy(fundamentals, sector=sector)
    b = buffett_strategy(fundamentals, sector=sector)
    gr = greenblatt_strategy(fundamentals, sector=sector)
    m = munger_strategy(fundamentals, sector=sector)
    l = lynch_strategy(fundamentals, growth, sector=sector)  # growth required
    s = schloss_strategy(fundamentals, sector=sector)
    d = davis_strategy(fundamentals, sector=sector)
    t = templeton_strategy(fundamentals, sector=sector)
    k = klarman_strategy(fundamentals, sector=sector)

    # ----------------------------
    # Quality & Forensics (Period-based)
    # ----------------------------
    piot = piotroski_f_score(income, balance, cashflow, sector=sector)
    bene = beneish_penalty(income, balance, cashflow, sector=sector)
    mont = montier_penalty(income, balance, cashflow, sector=sector)

    # ----------------------------
    # Gewichtete, konfigurationsbasierte Aggregation (no magic numbers)
    # ----------------------------
    value = combined_value_score(
        g, b, gr, m, l, s, d, t, k, piot, bene, mont,
        sector=sector
    )

    # ----------------------------
    # Response (UI ohne Sonderfälle)
    # ----------------------------
    return jsonify(
        symbol=symbol,
        sector=sector,
        fundamentals=fundamentals,
        growth=growth,

        # einzelne Strategien
        graham=g,
        buffett=b,
        greenblatt=gr,
        munger=m,
        lynch=l,
        schloss=s,
        davis=d,
        templeton=t,
        klarman=k,

        # Gesamtbewertung
        value=value,

        # Quality/Forensics Block
        quality_forensics={
            "piotroski": piot,
            "beneish": bene,
            "montier": mont,
        },
    )



@app.route("/bundle", methods=["GET"])
def bundle():
    symbol = request.args.get("symbol", "").upper().strip()
    sector = request.args.get("sector")  # optional

    fundamentals = data_layer.get_basic_fundamentals(symbol)
    growth = data_layer.get_growth(symbol)
    income = data_layer.get_income(symbol)
    balance = data_layer.get_balance(symbol)
    cashflow = data_layer.get_cashflow(symbol)

    return jsonify(run_default_value_bundle(
        fundamentals=fundamentals,
        growth=growth,
        income=income,
        balance=balance,
        cashflow=cashflow,
        sector=sector
    ))








