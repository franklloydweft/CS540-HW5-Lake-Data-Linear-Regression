import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    file = sys.argv[1]
    data = pd.read_csv(file)
    plot(data)
    years = np.asarray(data["year"]).astype(int)
    ones = np.ones(len(years)).astype(int)
    data_matrix = np.column_stack((ones, years))

    # suppressing scientific notation
    np.set_printoptions(suppress=True)

    print("Q3a:")
    print(data_matrix)

    print("Q3b:")
    days = np.asarray(data["days"])
    print(days)

    print("Q3c:")
    product = np.matmul(data_matrix.transpose(), data_matrix)
    print(product)

    # reset scientific notation
    np.set_printoptions(suppress=False)

    print("Q3d:")
    inverse_product = np.linalg.inv(product)
    print(inverse_product)

    print("Q3e:")
    pseudo_inverse = np.matmul(inverse_product, data_matrix.transpose())
    print(pseudo_inverse)

    print("Q3f:")
    beta_hat = np.matmul(pseudo_inverse, days)
    print(beta_hat)

    print("Q4: ", end="")
    y_hat_test = beta_hat[0] + (beta_hat[1] * 2022)
    print(y_hat_test)

    print("Q5a: ", end="")
    if y_hat_test > 0:
        print(">")
    elif y_hat_test < 0:
        print("<")
    else:
        print("=")

    print("Q5b: Towards the end of our time frame there looks to be a small uptick in slope graphically, "
          "\nso a slightly positive slope for a prediction here would make sense.")

    print("Q6a: ", end="")
    no_freeze = -beta_hat[0] / beta_hat[1]
    print(no_freeze)

    print("Q6b: The year 2455 makes sense to me, as the graph of data oscillates quite a bit despite its overall\n "
          "negative trend. So, from intuition I wouldn't estimate a year right around the corner, but I also wouldn't\n"
          " rule out a timeframe as small (for climate changes) as ~430 years.")


def plot(data):
    plt.figure()
    data.plot(x="year", y="days", legend=None)
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.savefig("plot.jpg")


if __name__ == '__main__':
    main()
