# CS540 HW5 Spring 2023

## Required Packages
- NumPy `import numpy as np`
- Pandas `import pandas as pd`
- matplotlib `import matplotlib.pyplot as plt`

## Dataset
Record of number of days Lake Mendota is covered by ice kept by The University of Wisconsin - Madison. Part of the assignment was cleaning the data into an organized csv file with ice data starting at year 1855/56 and ending at year 2021/2022. This cleaned data is included in this repository.

## Processes
1. Data visualization performed in the function `plot(data)` where data is a read CSV file. Every operation done on the data is plotted with this function using matplotlib functions.
2. First, a linear regression model is trained using the entire dataset as training set. This is done piecewise and steps are demarcated with print statements. Q3A: representation of data in matrix form with created feature vectors as rows, Q3B: creation of `days` vector, Q3C: matrix product of transpose data matrix and itself, Q3D: computation of inverse matrix product, Q3E: Computation of "pseudo-inverse," or inverse of matrix product from earlier multiplied by transpose data matrix, and finally, Q3F: compute closed form MLE (maximum likelihood estimation) with combined info (multiplication of pseudo-inverse and `days` vector).
3. Prediction of number of ice days for winter 2022-23 with computed MLE ("beta hat") with a "y hat test" and interpretation of that prediction
4. Prediction of year when lake no longer freezes and discussion of model limitation

*sign of beta hat 1 was incorrect in this assignment, leading to some slightly off assumptions*

