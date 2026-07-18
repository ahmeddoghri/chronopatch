from chronopatch import conformal_forecast, make_series

series = make_series()
forecast = conformal_forecast(series[:170], series[170:226])
print(f"next={forecast.point[0]:.2f} interval=({forecast.lower[0]:.2f}, {forecast.upper[0]:.2f})")
