# Tarzan v2.0

Motore di analisi portafoglio production-ready con arricchimento dati di mercato in tempo reale e dashboard Excel professionale.

## Architettura

```
tarzan/
├── __init__.py                  # Package root, versioning
├── main.py                      # CLI entry point (pipeline orchestrator)
├── config/
│   ├── __init__.py              # Configuration loader (YAML → typed accessors)
│   ├── constants.yaml           # Tunable parameters (risk-free rate, benchmarks, etc.)
│   └── static.yaml              # Rarely-changed mappings (exchanges, colors, etc.)
├── models/
│   ├── __init__.py              # Model exports
│   ├── holding.py               # Holding dataclass, AssetClass/Geography enums
│   ├── investor_config.py       # InvestorConfig with CSV deserialization
│   └── portfolio.py             # PortfolioMetrics (output DTO)
├── utils/
│   ├── __init__.py              # Utility module docs
│   ├── exceptions.py            # Domain-specific exception hierarchy
│   ├── holdings_loader.py       # Data Ingestion: CSV/XLSX → list[Holding]
│   ├── config_loader.py         # Config loading facade
│   ├── data_fetcher.py          # Data Enrichment: yfinance, OpenFIGI, FX, classification
│   ├── geo_scraper.py           # Geographic allocation: justETF + MSCI factsheets
│   ├── calculators.py           # Metric Computation: performance, risk, allocations
│   └── excel_generator.py       # Reporting: 8-sheet Excel dashboard
├── data/                        # Sample input files
│   ├── config_sample.csv
│   └── holdings_fineco_021726.csv
├── tests/
│   ├── __init__.py
│   └── conftest.py
├── output/                      # Generated dashboards and logs
└── requirements.txt
```

## Installazione

```bash
pip install -r requirements.txt
```

## Utilizzo

```bash
# Base (con defaults)
python -m tarzan.main --input_holdings data/holdings_fineco_021726.csv

# Completo
python -m tarzan.main \
    --input_holdings data/holdings_fineco_021726.csv \
    --input_config data/config_sample.csv \
    --output output/
```

## Input

### Holdings (obbligatorio)

`.xlsx` o `.csv` con colonne (case-insensitive):

| Colonna | Tipo | Obbligatorio | Descrizione |
|---------|------|:---:|-------------|
| isin | str | ✓ | Codice ISIN a 12 caratteri |
| ticker | str | ✓ | Ticker Yahoo Finance |
| quantity | float | ✓ | Numero di unità (>0) |
| cost_basis_eur | float | ✓ | Costo totale in EUR |
| market_value_eur | float | ✓ | Valore di mercato in EUR |
| currency | str | ✓ | Valuta dello strumento |
| usa, japan, eurozone_emu, ... | float | | Breakdown geografico (%) |

### Config (opzionale)

`config.csv` con coppie chiave-valore:

| Chiave | Default | Descrizione |
|--------|---------|-------------|
| monthly_invest_capacity | 0 | Budget mensile EUR |
| geo_exposure | 20% ciascuno | JSON: target allocazione geografica |
| allocation_targets | Equity 65%... | JSON: target asset class |
| rebalancing_threshold | 5.0 | Soglia ribilanciamento (%) |

## Metriche Finanziarie

### Performance
- CAGR, YTD, rendimenti periodici (1d–5y), IRR

### Risk (cutting-edge)
- **Sharpe Ratio**: rendimento risk-adjusted (excess return / volatilità)
- **Sortino Ratio**: penalizza solo la volatilità al ribasso
- **Max Drawdown**: massima perdita peak-to-trough
- **VaR (95%)**: Value at Risk via simulazione storica (non-parametrico)
- **CVaR (95%)**: Expected Shortfall — media delle perdite oltre il VaR (misura coerente di rischio, Artzner et al. 1999)
- **Volatilità Realizzata**: rolling window annualizzata
- **Beta / Alpha**: CAPM vs S&P 500

### Allocazioni
- Per asset class, geografia (solo equity), settore
- Supporto ETF multi-geografia con split proporzionale
- Delta vs target con suggerimenti di ribilanciamento

### Benchmark
- Confronto con 20+ benchmark (S&P 500, ACWI, VTI, AVUV, etc.)
- Analisi what-if: valore ipotetico se investito in ciascun benchmark

## Output

`portfolio_dashboard_[YYYYMMDD].xlsx` con 8 fogli:

1. **Dashboard** — KPI (inclusi VaR/CVaR), donut chart, top/bottom performers, goals
2. **Holdings** — Tabella completa arricchita con fonti dati e timestamp
3. **Allocations** — Pie/bar chart con actual vs target
4. **Performance** — Rendimenti cumulativi, griglie per periodo, overlay holdings
5. **Risk** — Metriche di rischio complete, drawdown chart, scatter risk-return
6. **Multi-Purpose Analysis** — Contribuzione al rendimento, breakdown, azioni di ribilanciamento
7. **Benchmark** — Confronto vs 20+ benchmark, performance cumulativa, what-if
8. **Documentation** — Descrizione e formula di ogni metrica

## Principali Migliorie (v1 → v2)

1. **Metriche cutting-edge**: aggiunta VaR (95%), CVaR/Expected Shortfall (95%), volatilità realizzata rolling
2. **Gerarchia eccezioni**: `TarzanError` con sottoclassi specifiche per dati finanziari
3. **Principi SOLID/DRY**: funzioni estratte e riutilizzabili, classificazione modulare, zero duplicazione
4. **Documentazione Google-style**: ogni funzione documentata con Args/Returns/Raises
5. **Commenti matematici**: spiegazione del "perché" dietro VaR storico vs parametrico, CVaR come misura coerente
6. **Error handling robusto**: isolamento errori per-holding, fallback graceful, logging granulare
7. **Output serializzabile**: `PortfolioMetrics.to_summary_dict()` per integrazione API/pipeline
8. **Dead code rimosso**: eliminati parametri ridondanti, variabili inutilizzate, import superflui
9. **Type safety**: annotazioni complete, guard clause per None, dict tipizzati
10. **Modello Holding arricchito**: metodi `is_enriched()` e `unrealized_gain_eur()`

## Struttura Futura Consigliata

```
tarzan/
├── core/                    # Logica di dominio pura (no I/O)
│   ├── risk_engine.py       # Motore di rischio standalone (VaR, CVaR, stress test)
│   ├── optimizer.py         # Mean-variance optimization (Markowitz)
│   └── scenario_engine.py   # Monte Carlo simulation, stress testing
├── adapters/                # Interfacce verso sistemi esterni
│   ├── yfinance_adapter.py  # Astrazione su yfinance
│   ├── openfigi_adapter.py  # Astrazione su OpenFIGI
│   └── bloomberg_adapter.py # Futuro: Bloomberg Terminal API
├── api/                     # REST API layer
│   └── fastapi_app.py       # Endpoint per integrazione pipeline
├── streaming/               # Real-time data
│   └── websocket_feed.py    # Live price updates
└── tests/
    ├── unit/
    ├── integration/
    └── property/             # Hypothesis-based property tests
```
