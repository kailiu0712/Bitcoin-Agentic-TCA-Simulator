1. Do we need to consider execution cost, or just impact?


3. Simple heuristic rule-based liquidation agents seem to outperform rl agents in cost and impact.

4. Hard part is still calibration and agent setup. Need to carefully consider what rules needs to be tuned and how complicated the agent framework should be. 

5. Rules that are easy to tune: return dist., volatility, ACF, depth dist., trade freq.
Rules hard to tune: size impact curve, flow impact curve, response curve

6. How to measure the accuracy of our simulated market impact? It seems difficult to validate with real data, without the simulation setup.

7. how long data  /  stress data


