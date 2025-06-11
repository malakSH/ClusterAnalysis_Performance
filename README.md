This folder contains synthetic datasets generated for evaluating clustering algorithms. These datasets are designed to provide controlled scenarios for testing algorithm performance, scalability, and robustness. It is used for:

Benchmarking clustering algorithms;
Visualising clustering performance;
Comparing and interpreting runtime and scalability for different sizes Here are the input variables used to generate the datasets using Gaussian Distribution formula <https://numpy.org/doc/stable/reference/random/generated/numpy.random>. normal.html , with n being the number of instances, m being the number of features and c being the number of cluster centers:
| Data | Label | n | m | c |

| 1 | 100 | 5 | 2 |

| 2 | 1000 | 10 | 4 |

| 3 | 2000 | 15 | 6 |

| 4 | 3000 | 20 | 8 |

| 5 | 4000 | 25 | 10 |

| 6 | 5000 | 30 | 12 |

| 7 | 6000 | 35 | 14 |

| 8 | 7000 | 40 | 16 |

| 9 | 8000 | 45 | 18 |

| 10 | 9000 | 50 | 20 |

| 11 | 10000 | 55 | 22 |
