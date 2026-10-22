# CALM liquidation replay dashboard

CALM: Coinbase Agent-based Liquidation & Market Simulator. Runs a market model (Coinbase SOL-USD,
six models) twice with the same seed, with and without a TWAP liquidation, and replays both side by side.

<video src="sim-demo.mp4" controls width="100%"></video>

[[▶ Watch the demo (sim-demo.mp4)](sim-demo.mp4)](https://github.com/user-attachments/assets/df67aa13-8f6e-45fc-bd25-2cb2b357e1fe)

## Quick run

```
pip install -r requirements.txt       # numpy, pandas, tqdm (Python 3.12+)
python server.py
```

Open http://localhost:8050, choose a model and press **Run** (or open `http://localhost:8050/?autorun`).
The default run of every model is cached and replays at once; Space plays / pauses.


## License and data

Add an explicit license before public redistribution if required. Market data are not
included; users are responsible for data licensing and exchange terms.
