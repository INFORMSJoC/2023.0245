# ADP Performance Evaluation

This folder contains the source code for evaluating the performance of ADP for different values of maximum shelf-life $m$.

## Overview

The evaluation involves different approaches depending on the value of the maximum shelf-life parameter $m$. These methods are outlined below:

- ### $m = 3$:  
  For a maximum shelf-life of three days, we evaluate the performance of both ADP and the lower-bound against the optimal policy to assess their accuracy.

- ### $m = 5$:  
  For a maximum shelf-life of five days, the optimal policy becomes computationally infeasible to calculate. Instead, we compare the ADP performance with the lower-bound.

- ### $m = 8$:   
  For a maximum shelf-life of eight days, even the lower-bound becomes computationally intractable. Here, the ADP performance is compared to the initial Myopic policy.
