# Predictive Maintenance: Remaining Useful Life Estimation with Uncertainty Bounds

Predicting how many operating cycles remain before a turbofan engine fails and quantifying the confidence in that prediction on the NASA C-MAPSS (FD001) dataset.

**Objective:** Predict RUL with a calibrated 90% confidence interval, so maintenance engineers can act on both the estimate and its uncertainty. A baseline Random Forest is extended to an LSTM to capture sequence history, and the two are compared on RMSE and NASA Score. An LSTM with quantile regression then produces a 90% confidence bound for actionable predictions

## Results

| Model | Input context | Test RMSE | 90% CI Coverage | NASA Score |
|---|---|---|---|---|
| Random Forest | 1-cycle snapshot | 50.41 | — | 187925 |
| LSTM (point) | 30-cycle window | 13.80 | — | 352 |
| LSTM-Quantile (p50) | 30-cycle window | 14.48 | 93.6% | 460 |

*NASA score penalizes late predictions ~10× more than early ones, reflecting the real operational cost of a missed failure. Lower is better.*

## Takeaways

- The LSTM's 30-cycle window substantially beats the single-snapshot Random Forest baseline.
- Quantile LSTM matches the point model's accuracy while producing calibrated intervals.
- Model hits close to the 90% target overall, but tighter in the long-life plateau and looser in the mid-life degradation window — where it matters most. Stated plainly rather than averaged away.

## Approach

**Data & feature selection.** 21 sensors + 3 operational settings per cycle. Ten constant / near-constant channels carry no degradation signal and are dropped, reducing noise with no information loss.

**RUL labeling.** Targets are capped with a piecewise-linear function at 125 cycles — engines look healthy and indistinguishable early in life, so learning capacity is concentrated on the critical near-failure window.

**Models.**

1. *Random Forest* — baseline on the last observed cycle.
2. *LSTM (point estimate)* — sequence model on 30-cycle rolling windows.
3. *LSTM + quantile regression* — adds p5 / p50 / p95 outputs, trained with a pinball loss plus an MSE anchor on the median to prevent quantile collapse. Quantiles are sorted in the forward pass to enforce p5 ≤ p50 ≤ p95.


## Reproducing

```bash
pip install -r requirements.txt
# Download the FD001 data into data/ — see data/README.md
jupyter notebook notebooks/cmapss.ipynb
```


## Limitations & next steps

- FD001 is a single operating condition and fault mode. FD002–FD004 add multi-condition complexity and would test generalization.
- Production deployment would stream live sensor data through a 30-cycle buffer and emit p50 RUL plus an alert flag in real time.
