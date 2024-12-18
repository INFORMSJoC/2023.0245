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

The goal of this software is to provide an intelligent decision-support tool for managing perishable inventory systems, with a focus on blood platelet ordering. By leveraging a stochastic inventory control framework, the software addresses the challenges posed by uncertain shelf-life and fluctuating daily demand. It implements an Approximate Dynamic Programming (ADP) approach to approximate optimal ordering policies, significantly reducing expiration and shortage rates compared to historical benchmarks. The tool is designed for scalability and practical application, enabling healthcare providers to optimize inventory decisions efficiently, even under complex and uncertain conditions.


## Results

[Figure 1](results/Figure 1.pdf) in the paper compares the expected cost function, value function, and optimal policy obtained under the fixed
ordering cost of $κ = 10$ and endogenous shelf-life uncertainty (left column) with those obtained under
zero fixed ordering cost and endogenous (middle column) or deterministic shelf-life (right column)
assuming $h = 1, \, l = 20, \, θ = 5$ and the demand and other parameters are the same across columns.

![Figure 1](results/Figure 1.pdf)


## Replicating

To replicate the results in [Figure 1](results/Figure 1.pdf), do either


## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/mhdmjli).

## Support

For support in using this software, please contact Mahdi Mirjalili via
[email](mhdmjli@mie.utoronto.ca).
