# Lecture 2 extraction benchmark

Efficiency and numerical diagnostics only; this is not a downstream accuracy ranking.

| Dataset | Method | Status | Median seconds | Scalar columns | Reproducible |
|---|---|---|---:|---:|---|
| GunPoint | summary | ok | 0.247515 | 7 | True |
| GunPoint | acf | ok | 0.195159 | 10 | True |
| GunPoint | pacf | ok | 0.228490 | 10 | True |
| GunPoint | adf | ok | 0.311434 | 3 | True |
| GunPoint | permutation_entropy | ok | 0.049719 | 1 | True |
| GunPoint | spectrum | ok | 0.049476 | 5 | True |
| GunPoint | catch22 | partial | 0.230680 | 22 | True |
| GunPoint | tsfresh | partial | 5.343215 | 777 | True |
| ArrowHead | summary | ok | 0.268976 | 7 | True |
| ArrowHead | acf | ok | 0.211943 | 10 | True |
| ArrowHead | pacf | ok | 0.248452 | 10 | True |
| ArrowHead | adf | ok | 0.436565 | 3 | True |
| ArrowHead | permutation_entropy | ok | 0.068794 | 1 | True |
| ArrowHead | spectrum | ok | 0.057680 | 5 | True |
| ArrowHead | catch22 | partial | 0.268049 | 22 | True |
| ArrowHead | tsfresh | partial | 6.020362 | 777 | True |
| ItalyPowerDemand | summary | ok | 1.325084 | 7 | True |
| ItalyPowerDemand | acf | ok | 1.021520 | 10 | True |
| ItalyPowerDemand | pacf | ok | 1.201664 | 10 | True |
| ItalyPowerDemand | adf | ok | 1.069940 | 3 | True |
| ItalyPowerDemand | permutation_entropy | ok | 0.206902 | 1 | True |
| ItalyPowerDemand | spectrum | ok | 0.249422 | 5 | True |
| ItalyPowerDemand | catch22 | partial | 1.287234 | 22 | True |
| ItalyPowerDemand | tsfresh | partial | 25.215190 | 777 | True |
| BasicMotions | summary | ok | 0.552270 | 42 | True |
| BasicMotions | acf | ok | 0.446841 | 60 | True |
| BasicMotions | pacf | ok | 0.516244 | 60 | True |
| BasicMotions | adf | ok | 0.594580 | 18 | True |
| BasicMotions | permutation_entropy | ok | 0.078039 | 6 | True |
| BasicMotions | spectrum | ok | 0.068245 | 30 | True |
| BasicMotions | catch22 | partial | 0.695715 | 132 | True |
| BasicMotions | tsfresh | partial | 12.385142 | 4662 | True |
| Airline | summary | ok | 0.002624 | 7 | True |
| Airline | acf | ok | 0.002131 | 10 | True |
| Airline | pacf | ok | 0.002419 | 10 | True |
| Airline | adf | ok | 0.003118 | 3 | True |
| Airline | permutation_entropy | ok | 0.001505 | 1 | True |
| Airline | spectrum | ok | 0.001332 | 5 | True |
| Airline | catch22 | partial | 0.013903 | 22 | True |
| Airline | tsfresh | partial | 0.031592 | 782 | True |
| DailyDelhiClimate | summary | ok | 0.006406 | 28 | True |
| DailyDelhiClimate | acf | ok | 0.005155 | 40 | True |
| DailyDelhiClimate | pacf | ok | 0.005925 | 40 | True |
| DailyDelhiClimate | adf | ok | 0.006826 | 12 | True |
| DailyDelhiClimate | permutation_entropy | ok | 0.001789 | 4 | True |
| DailyDelhiClimate | spectrum | ok | 0.001808 | 20 | True |
| DailyDelhiClimate | catch22 | partial | 0.030209 | 88 | True |
| DailyDelhiClimate | tsfresh | partial | 0.117043 | 3128 | True |

See benchmark.json for load/write times, all warmup/repetition timings, failure reasons, parameters and environment.
Each dataset/method directory contains the first measured extraction, including nonfinite outputs.
