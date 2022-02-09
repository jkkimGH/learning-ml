# Learn enough about NumPy and pandas to understand tf.keras code.
# Learn how to use Colabs.
# Become familiar with linear regression code in tf.keras.
# Evaluate loss curves.
# Tune hyperparameters.

# Need to know a bit of NumPy and pandas

# NumPy Ultraquick Tutorial (just for this lesson's purpose)
import numpy as np

one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])

two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])

sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)

random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))

random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1)

# Broadcasting - easy way to avoid strict rules of linear algebra and do convenient manipulations to the matrices/vectors
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0

random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3

# Task 1: Create a Linear Dataset
# 1. Assign a sequence of integers from 6 to 20 (inclusive) to a NumPy array named feature.
# 2. Assign 15 values to a NumPy array named label such that:

feature = np.arange(6, 21)
print(feature)
label = (feature * 3) + 4
print(label)

# Task 2: Add Some Noise to the Dataset
# To make your dataset a little more realistic, insert a little random noise into each element of the label array you already created. 
# To be more precise, modify each value assigned to label by adding a different random floating-point value between -2 and +2.
# Don't rely on broadcasting. Instead, create a noise array having the same dimension as label.

noise = (np.random.random([15]) * 4) - 2
print(noise)
label = label + noise 
print(label)

# Pandas DataFrames Ultraquick Tutorial (just for this lesson's purpose)

# A DataFrame is similar to an in-memory spreadsheet. Like a spreadsheet:
  # A DataFrame stores data in cells.
  # A DataFrame has named columns (usually) and numbered rows.

import numpy as np
import pandas as pd

# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)

# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)

# Pandas provide multiples ways to isolate specific rows, columns, slices or cells in a DataFrame. 
print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])

# Task 1: Create a DataFrame

# Create a Python list that holds the names of the four columns.
my_column_names = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

# Create a 3x4 numpy array, each cell populated with a random integer.
my_data = np.random.randint(low=0, high=101, size=(3, 4))

# Create a DataFrame.
df = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(df)

# Print the value in row #1 of the Eleanor column.
print("\nSecond row of the Eleanor column: %d\n" % df['Eleanor'][1])

# Create a column named Janet whose contents are the sum
# of two other columns.
df['Janet'] = df['Tahani'] + df['Jason']

# Print the enhanced DataFrame
print(df)

# Copying a DataFrame
# Pandas provides 2 ways: 
  # Referencing. If you assign a DataFrame to a new variable, any change to the DataFrame or to the new variable will be reflected in the other.
  # Copying. If you call the pd.DataFrame.copy method, you create a true independent copy. 
    # Changes to the original DataFrame or to the copy will not be reflected in the other.

# Referencing
ref_copy = df

# Copying
independent_copy = df.copy()
