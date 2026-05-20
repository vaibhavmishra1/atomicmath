# atomicmath docs

`atomicmath` now has one primary path: iterative lineage synthesis.

The runner samples parent rows from the latest accepted dataset iteration,
creates the next iteration only, records the full artifact and memory trail in
the generated rows, and optionally publishes those rows back to the Hub.

The implementation lives in:

- `atomicmath/lineage.py`
- `atomicmath/config.py`
- `atomicmath/cli.py`

Run it with:

```bash
python3 -m atomicmath.cli run --config examples/config.example.yaml
```
