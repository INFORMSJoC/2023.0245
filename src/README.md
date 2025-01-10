# ADP Performance Evaluation

This folder contains the source code for evaluating the performance of ADP for different values of maximum shelf-life $m$. We use parallel processing to run multiple parameter combinations simultaneously on the [Niagara cluster](https://docs.alliancecan.ca/wiki/Niagara).

## Overview

The evaluation involves different approaches depending on the value of the maximum shelf-life parameter $m$. These methods are outlined below:

- ### $m = 3$:  
  For a maximum shelf-life of three days, we evaluate the performance of both ADP and the lower-bound against the optimal policy to assess their accuracy.

- ### $m = 5$:  
  For a maximum shelf-life of five days, calculating the optimal policy becomes computationally infeasible. Instead, we compare the ADP performance to the lower-bound and other policies, including the exact policy under a deterministic maximum shelf-life of five days in the case study.

- ### $m = 8$:   
  For a maximum shelf-life of eight days, even the lower-bound becomes computationally intractable. Here, the ADP performance is compared to the initial Myopic policy.
