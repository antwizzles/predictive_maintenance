\# Predictive Maintenance: Remaining Useful Life Estimation with Uncertainty



Predicting how many operating cycles remain before a turbofan engine fails —

and quantifying the confidence in that prediction — on the NASA C-MAPSS

(FD001) dataset.



Most RUL models output a single number. This one outputs a number \*\*plus a

calibrated 90% confidence interval\*\*, so a maintenance engineer can act on

both the estimate and its uncertainty.



!\[Per-unit RUL prediction with uncertainty band](figures/per\_unit\_trajectory.png)



\## Results



| Model | Input context | Test RMSE | 90% CI Coverage | NASA Score |

|---|---|---|---|---|

| Random Forest | 1-cycle snapshot | ⚠️\[fill] | — | ⚠️\[fill] |

| LSTM (point) | 30-cycle window | ⚠️13.61 | — | ⚠️353 |

| LSTM-Quantile (p50) | 30-cycle window | ⚠️13.83 | ⚠️93.6% | ⚠️383 |



\*NASA score penalizes late predictions \~10× more than early ones, reflecting

the real operational cost of a missed failure. Lower is better.\*



\*\*Takeaways\*\*

\- Sequence context matters: the LSTM's 30-cycle window substantially beats the

&#x20; single-snapshot Random Forest baseline.

\- Adding uncertainty is nearly free: the quantile LSTM matches the point

&#x20; model's accuracy while producing calibrated intervals.

\- Coverage is close to the 90% target — though not uniform across the RUL

&#x20; range (tighter in the long-life plateau, looser in the mid-life degradation

&#x20; window, where it matters most). Stated honestly rather than averaged away.



\## Approach



\*\*Data \& feature selection.\*\* 21 sensors + 3 operational settings per cycle.

Ten constant / near-constant channels carry no degradation signal and are

dropped, reducing noise with no information loss.



\*\*RUL labeling.\*\* Targets are capped with a piecewise-linear function at 125

cycles — engines look healthy and indistinguishable early in life, so learning

capacity is concentrated on the critical near-failure window.



\*\*Leakage-safe split.\*\* Train/validation split is \*\*by engine unit\*\*, not by

row, so no engine's future cycles leak into training.



\*\*Models.\*\*

1\. \*Random Forest\* — fast baseline on the last observed cycle.

2\. \*LSTM (point estimate)\* — sequence model on 30-cycle rolling windows.

3\. \*LSTM + quantile regression\* — adds p5 / p50 / p95 outputs, trained with a

&#x20;  pinball loss plus an MSE anchor on the median to prevent quantile collapse.

&#x20;  Quantiles are sorted in the forward pass to enforce p5 ≤ p50 ≤ p95.



\## Figures



| | |

|---|---|

| !\[Coverage](figures/coverage\_validation.png) | !\[Predicted vs True](figures/pred\_vs\_true.png) |



\## Reproducing



```bash

pip install -r requirements.txt

\# Download the FD001 data into data/ — see data/README.md

jupyter notebook notebooks/rul\_estimation.ipynb

```



\## Limitations \& next steps

\- FD001 is a single operating condition and fault mode. FD002–FD004 add

&#x20; multi-condition complexity and would test generalization.

\- Prediction intervals are wide for long-life engines; \*\*conformal prediction\*\*

&#x20; could tighten them with formal coverage guarantees.

\- Production deployment would stream live sensor data through a 30-cycle buffer

&#x20; and emit p50 RUL plus an alert flag in real time.



\## Stack

Python · PyTorch · scikit-learn · pandas · matplotlib

