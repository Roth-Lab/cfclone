# TODO

### Next time
- Add `plot_fit` and `plot_clone` to cli.
- Implement `plot_mcmc_trace` and `plot_laplace_trace`.
- Get rid of `arviz` dependency by implementing HDI intervals from scratch. 

### Testing
- `pixi run -e test pytest tests/test_postprocess/test_write_parameter_summaries.py`
- `pixi run -e test pytest tests/test_postprocess/test_write_posterior_predictive.py`