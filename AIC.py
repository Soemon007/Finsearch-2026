import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from feature_engineering import series
warnings.filterwarnings("ignore", category=ConvergenceWarning)

best_aic = float("inf")
best_order = None
best_model = None

for p in range(4):
    for q in range(4):

        try:
            model = ARIMA(series, order=(p, 0, q))
            result = model.fit()

            print(f"ARIMA({p},0,{q}) -> AIC = {result.aic:.2f}")

            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, 0, q)
                best_model = result

        except Exception as e:
            print(f"ARIMA({p},0,{q}) failed: {e}")

print("Best Order :", best_order)
print("Best AIC   :", best_aic)

if best_model is not None:
    print(best_model.summary())