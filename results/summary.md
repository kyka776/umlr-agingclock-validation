# UMLR aging-clock audit: summary

## Русский

### Итог

Gate A — **GO только для независимого аудита**: официальный код найден, поэтому
второй production-package не создавался. Gate B — **GO**: формулы, ограничения,
отсутствие test-refit и известные simulation properties подтверждены тестами.
Gate C — **CONDITIONAL**: UMLR нельзя рекомендовать по одному снижению bias.

В 30 воспроизведениях каждого из десяти заранее заданных сценариев UMLR уменьшал
абсолютный residual-age slope во всех сценариях с bias. Но его RMSE был выше raw
Lasso в 8 из 10 сценариев: в regression-to-the-mean на 30%, при age imbalance на
35%, в heteroscedastic, null-outcome, latent-signal, confounding, small-n и
nonlinear сценариях — примерно на 8–10%. Улучшение RMSE было только в почти
идеальном сценарии и при range shift.

В сценариях с известным latent signal UMLR обычно лучше восстанавливал
age-adjusted association и покрывал истинную цель доверительным интервалом.
Исключения важны: при null outcome и nonlinear stress test ошибка association
слегка выросла. Residual correction почти обнулял residual-age slope и снижал
RMSE, но после корректировки на chronological age не восстанавливал потерянный
outcome signal: это следует из самой линейной трансформации, а не из бага.

Открытый OmniAge lung tutorial cohort использован без подбора результата:
56 образцов, 2 974 признака, фиксированный stratified split 38/18, feature
selection и scaling только на train. На test MAE составил:

| Метод | MAE, лет | RMSE, лет | residual-age slope |
|---|---:|---:|---:|
| Raw Lasso | 10,2 | 13,0 | -1,19 |
| Residual correction | 7,1 | 9,8 | -0,67 |
| Linear recalibration | 10,5 | 13,4 | -1,20 |
| UMLR oracle | 12,1 | 15,9 | -1,23 |

Все disease-state confidence intervals широки и включают ноль. Это маленький
exploratory holdout, а не клиническая проверка. Он не подтверждает переносимость
UMLR и честно служит falsification case.

### Когда применять нельзя

- нет независимого calibration/training cohort с тем же возрастным диапазоном;
- correction пришлось бы оценивать на test ages;
- важнее ошибка индивидуального возраста, а рост RMSE не приемлем;
- после range shift, смены ткани/платформы или feature pipeline нет отдельной
  внешней проверки;
- вывод основан только на flat residual-age slope;
- планируется клиническая интерпретация без соответствующего validation study.

### Практическая рекомендация

UMLR стоит рассматривать как training-time constraint для конкретного
downstream-inference use case. До применения нужно заранее зафиксировать raw
baseline, calibration metrics, MAE/RMSE, outcome association target и критерий
допустимого ухудшения. Test set должен оставаться untouched.

## English

### Bottom line

Gate A is **GO for an independent audit only** because official code already
exists. Gate B is **GO**: equation, constraint, leakage-boundary, edge-case and
known-simulation tests pass. Gate C is **CONDITIONAL**: UMLR should not be
recommended from a single bias plot.

Across 30 replications of each of ten frozen scenarios, UMLR reduced absolute
residual-age slope in every biased scenario. Its chronological-age RMSE was
higher than raw Lasso in eight scenarios, including +30% under
regression-to-the-mean and +35% under age imbalance. It generally improved
recovery and interval inclusion for known latent outcome associations, but not
for the null-outcome and nonlinear stress tests.

The pre-specified open OmniAge holdout used 38 training and 18 untouched test
samples. UMLR had the highest holdout MAE (12.1 years); raw Lasso had 10.2 and
residual correction 7.1. All disease-state association intervals were wide and
crossed zero. This small example is exploratory falsification evidence, not a
clinical or definitive performance claim.

### Recommendation

Treat UMLR as a training-time constraint for a specified downstream-inference
problem. Use an external calibration cohort, keep the test set untouched, and
evaluate residual-age bias together with MAE/RMSE and outcome-association
recovery. Do not deploy it as a universal post-hoc clock correction.

## Evidence

- [`tradeoff_table.csv`](tables/tradeoff_table.csv)
- [`omniage_holdout_metrics.csv`](tables/omniage_holdout_metrics.csv)
- [`simulation-benchmark.png`](figures/simulation-benchmark.png)
- [`omniage-holdout.png`](figures/omniage-holdout.png)
- [`benchmark.ipynb`](../notebooks/benchmark.ipynb)
