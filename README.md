[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Platelet Inventory Management with Approximate Dynamic Programming

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Platelet Inventory Management with Approximate Dynamic Programming](https://doi.org/10.1287/ijoc.2023.0245) by Hossein Abouee-Mehrizi, Mahdi Mirjalili, and Vahid Sarhangian. 
The snapshot is based on 
[this SHA](https://github.com/tkralphs/JoCTemplate/commit/f7f30c63adbcb0811e5a133e1def696b74f3ba15) 
in the development repository. 

**Important: This code is being developed on an on-going basis at 
https://github.com/tkralphs/JoCTemplate. Please go there if you would like to
get a more recent version or would like support**

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0245

https://doi.org/10.1287/ijoc.2023.0245.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{PLTADP,
  author =        {Abouee-Mehrizi, Hossein and Mirjalili, Mahdi and Sarhangian, Vahid},
  publisher =     {INFORMS Journal on Computing},
  title =         {Platelet Inventory Management with Approximate Dynamic Programming},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0245.cd},
  url =           {https://github.com/INFORMSJoC/2023.0245},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0245},
}  
```

## Description

This repository provides the code and data requried to reproduce the results of the experiments and case study reported in [Platelet Inventory Management with Approximate Dynamic Programming](https://doi.org/10.1287/ijoc.2023.0245). 


## Results

[Figure 1](results/Figure%201.pdf) in the paper compares the expected cost function, value function, and optimal policy obtained under the fixed
ordering cost of $κ = 10$ and endogenous shelf-life uncertainty (left column) with those obtained under
zero fixed ordering cost and endogenous (middle column) or deterministic shelf-life (right column)
assuming $h = 1, \, l = 20, \, θ = 5$ and the demand and other parameters are the same across columns.

[Figure 2 (A)](results/Figure%202%20(A).pdf) in the paper shows the effect of order size coefficients in the multinomial logistic model on the remaining shelf-life distribution
for order sizes of 0. The larger absolute magnitude of positive (negative) coefficients decreases (increases) the probability of receiving units with the remaining shelf-life of one day more rapidly.

[Figure 2 (B)](results/Figure%202%20(B).pdf) in the paper shows the effect of order size coefficients in the multinomial logistic model on the remaining shelf-life distribution
for order sizes of 5. The larger absolute magnitude of positive (negative) coefficients decreases (increases) the probability of receiving units with the remaining shelf-life of one day more rapidly.

[Figure 2 (C)](results/Figure%202%20(C).pdf) in the paper shows the effect of order size coefficients in the multinomial logistic model on the remaining shelf-life distribution
for order sizes of 10. The larger absolute magnitude of positive (negative) coefficients decreases (increases) the probability of receiving units with the remaining shelf-life of one day more rapidly.

[Figure 2 (D)](results/Figure%202%20(D).pdf) in the paper shows the effect of order size coefficients in the multinomial logistic model on the remaining shelf-life distribution
for order sizes of 15. The larger absolute magnitude of positive (negative) coefficients decreases (increases) the probability of receiving units with the remaining shelf-life of one day more rapidly.

[Figure 3](results/Figure%203.pdf) in the paper shows the performance of candidate basis functions with respect to their MAPE calculated using the optimal
value function.

[Figure 4](results/Figure%204.pdf) in the paper shows the relative optimality gap of the ADP policy for cases with a maximum shelf-life of $m=3$ days. The black line presents the estimate of the expected
optimality gap in each iteration and the gray area is the corresponding 95% confidence interval. On
average, the estimate of the expected optimality gap among all cases is 1.8%, indicating an 80%
reduction compared to 8.9% of the initial Myopic solution at iteration zero.

[Figure 5](results/Figure%205.pdf) in the paper shows the performance improvement for cases with more than 5% optimality gaps in Figure 4. The interaction
term $x_1x_2$ can further improve the results compared to the cubic terms $x^3_1$ and $x^3_2$.

[Figure 6](results/Figure%206.pdf) in the paper shows the quality of the lower-bound obtained using the imperfect information relaxation approach for cases with a maximum shelf-life of $m=3$ days. On average,
the relative gap between the lower-bound (red) and optimal (blue) is 4.9%.

[Figure 7](results/Figure%207.pdf) in the paper shows the performance of the algorithm for cases with a maximum shelf-life of $m=5$ days. The black line is the best estimate of the expected cost in each iteration
and the red line is the estimate of the expected cost with imperfect information relaxation. The shaded areas are the 95% confidence intervals. On average, the ADP policy can reduce the gap between the 
upper-bound, i.e., Myopic policy at iteration zero, and lower-bound from 16.2% to 6.6%.

[Figure 8](results/Figure%208.pdf) in the paper shows the impact of ignoring endogenous shelf-life uncertainty when $m = 3$. On average, the estimate of the expected optimality gap among all cases is 50.9% (red), 5.7% (orange), and 5.5% (green) for the
policies obtained under the deterministic shelf-life and exogenous distributions.

[Figure 9](results/Figure%209.pdf) in the paper shows the impact of ignoring endogenous shelf-life uncertainty when $m = 5$. On average, the estimate of the expected relative gap in cost compared with the ADP policy obtained under the true setting is 21.5%
(red), 6.1% (orange), and 3.6% (green) for the policies obtained under the deterministic shelf-life as well as Myopic, and ADP exogenous shelf-life uncertainty, respectively.

[Figure 10](results/Figure%2010.pdf) in the paper shows daily platelet demand at HGH. The red dashed line is the estimated average daily demand within the years 2015-2016. The blue line is the estimated average daily demand varying across days of the week.

[Figure 11](results/Figure%2011.pdf) in the paper compares average daily demand in 2017 with average inventory levels after receiving orders placed using Endog., Exoge., and Deter. policies obtained under different cost settings in Table 4.

[Figure EC.1](results/Figure%20EC.1.pdf) in the paper shows the performance of the algorithm for cases with a maximum shelf-life of $m=8$ days. The black line is the estimate of the expected relative reduction in cost
compared to the initial Myopic policy at iteration zero. The gray area is the 95% confidence interval. On average, the estimate of the expected relative reduction in cost among cases with $κ = 10$ and $κ= 100$ is 1.4% and 18.7%, respectively.

[Figure EC.2](results/Figure%20EC.2.pdf) in the paper shows the sensitivity to demand for cases with $m = 3$. On average among all tested scenarios, the estimate of the expected relative improvement in the upper-bound, i.e., Myopic policy at iteration zero, after
using the ADP approach is 4.9% and 4.3% for the real (red) and larger (blue) demand, respectively.

[Figure EC.3](results/Figure%20EC.3.pdf) in the paper shows the sensitivity to demand for cases with $m= 5$. On average among all tested scenarios, the estimate of the expected relative improvement in the upper-bound, i.e., Myopic policy at iteration zero, after
using the ADP approach is 6.8% and 7.6% for the real (red) and larger (blue) demand, respectively.

[Figure EC.4](results/Figure%20EC.4.PNG) in the paper compares the fitted negative binomial and Poisson distributions with historical demand data for Mondays $(τ = 0)$ during the years 2015 and 2016.

[Figure EC.5](results/Figure%20EC.5.PNG) in the paper compares the fitted negative binomial and Poisson distributions with historical demand data for Sundays $(τ = 6)$ during the years 2015 and 2016.

[Figure EC.6 (A)](results/Figure%20EC.6%20(A).pdf) in the paper compares the historical distribution of remaining shelf-life with the fitted multinomial distribution for order sizes of 6 at HGH in 2017.

[Figure EC.6 (B)](results/Figure%20EC.6%20(B).pdf) in the paper compares the historical distribution of remaining shelf-life with the fitted multinomial distribution for order sizes of 8 at HGH in 2017.

## Replicating

To replicate the results in [Figure 1](results/Figure%201.pdf), run the notebook [Figure 1.ipynb](scripts/Figure%201.ipynb). The pickle data required for this replication is available upon request and has not been uploaded to this repository.

To replicate the results in [Figure 2 (A)](results/Figure%202%20(A).pdf), [Figure 2 (B)](results/Figure%202%20(B).pdf), [Figure 2 (C)](results/Figure%202%20(C).pdf), and [Figure 2 (D)](results/Figure%202%20(D).pdf), run the notebook [Figure 2.ipynb](scripts/Figure%202.ipynb).

To replicate the results in [Figure 3](results/Figure%203.pdf), run the R script [Figure 3.R](scripts/Figure%203.R). The CSV data required for this replication is available in the [data](data/Figure%203/) folder.

To replicate the results in [Figure 4](results/Figure%204.pdf), run the R script [Figure 4.R](scripts/Figure%204.R). The CSV data required for this replication is available in the [data](data/Figure%204/) folder.

To replicate the results in [Figure 5](results/Figure%205.pdf), run the R script [Figure 5.R](scripts/Figure%205.R). The CSV data required for this replication is available in the [data](data/Figure%205/) folder.

To replicate the results in [Figure 6](results/Figure%206.pdf), run the R script [Figure 6.R](scripts/Figure%206.R). The CSV data required for this replication is available in the [data](data/Figure%206/) folder.

To replicate the results in [Figure 7](results/Figure%207.pdf), run the R script [Figure 7.R](scripts/Figure%207.R). The CSV data required for this replication is available in the [data](data/Figure%207/) folder.

To replicate the results in [Figure 8](results/Figure%208.pdf), run the R script [Figure 8.R](scripts/Figure%208.R). The CSV data required for this replication is available in the [data](data/Figure%208/) folder.

To replicate the results in [Figure 9](results/Figure%209.pdf), run the R script [Figure 9.R](scripts/Figure%209.R). The CSV data required for this replication is available in the [data](data/Figure%209/) folder.

To replicate the results in [Figure 10](results/Figure%2010.pdf), run the R script [Figure 10.R](scripts/Figure%2010.R). Due to data privacy and security reasons, the data has been simulated within the code using the fitted negative binomial distributions.

To replicate the results in [Figure 11](results/Figure%2011.pdf), run the R script [Figure 11.R](scripts/Figure%2011.R). The CSV data required for this replication is available in the [data](data/Figure%2011/) folder.

To replicate the results in [Figure EC.1](results/Figure%20EC.1.pdf), run the R script [Figure EC.1.R](scripts/Figure%20EC.1.R). The CSV data required for this replication is available in the [data](data/Figure%20EC.1/) folder.

To replicate the results in [Figure EC.2](results/Figure%20EC.2.pdf), run the R script [Figure EC.2.R](scripts/Figure%20EC.2.R). The CSV data required for this replication is available in the [data](data/Figure%20EC.2/) folder.

To replicate the results in [Figure EC.3](results/Figure%20EC.3.pdf), run the R script [Figure EC.3.R](scripts/Figure%20EC.3.R). The CSV data required for this replication is available in the [data](data/Figure%20EC.3/) folder.

To replicate the results in [Figure EC.4](results/Figure%20EC.4.PNG), run the R script [Figure EC.4.R](scripts/Figure%20EC.4.R). Due to data privacy and security reasons, the data has been simulated within the code using the fitted negative binomial distributions. 

To replicate the results in [Figure EC.5](results/Figure%20EC.5.PNG), run the R script [Figure EC.5.R](scripts/Figure%20EC.5.R). Due to data privacy and security reasons, the data has been simulated within the code using the fitted negative binomial distributions.

To replicate the results in [Figure EC.6 (A)](results/Figure%20EC.6%20(A).pdf) and [Figure EC.6 (B)](results/Figure%20EC.6%20(B).pdf), run the R script [Figure EC.6.R](scripts/Figure%20EC.6.R). The CSV data required for this replication is available in the [data](data/Figure%20EC.6/) folder.

To replicate the results in Table 4 of the paper, run the notebook [Table 4.ipynb](scripts/Table%204.ipynb). The required pickle data for this replication is available upon request and has not been uploaded to this repository. Additionally, due to data privacy and security concerns, the out-of-sample demand data has been simulated within the code using fitted negative binomial distributions.

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/mhdmjli).

## Support

For support in using this software, submit an
[issue](https://github.com/tkralphs/JoCTemplate/issues/new).
