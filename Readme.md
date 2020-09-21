# Applicant Task: Backtest Slices

## Directory Layout

- The `data` folder that contains the time series data in the `parquet` file format.
- The `backtesting` folder contains the `BacktestingSlices` class and the unit test class `TestBacktestingSlices`.
- The `notebook` folder includes a Jupyter notebook where one can examine the given time series and possibly examine the results of his implementation.


## What to do?

#### Step 1: Create Conda Environment

Create the `conda` environment. We recommend to install [miniconda](https://docs.conda.io/en/latest/miniconda.html), if you do not have an existing conda installation.

```
conda env create -f conda_env.yaml
```

Activate the installed `conda` environment.
```
conda activate env_applicant_task
```

Start Jupyter lab and examine the given time series (Optional):
```
jupyter lab
```

#### Step 2: Implement the two methods in the `BacktestingSlices` class

Use your favorite Python IDE for your implementation task. We are using the IDE [PyCharm](https://www.jetbrains.com/de-de/pycharm/) at DataZoo.


#### Step 3: Perform unit tests on your implementation


#### Step 4: Push your solution to your favorite repository

The original repository is used as the base repository for applicant tasks and no solutions should be pushed there. . Therefore, please create an own repository for your solution and push it there. 

#### Step 5: Clean your environment (optional)

```
conda remove -n env_applicant_task --all
```


