
Pandas is a powerful and widely used data manipulation library in Python. It provides data structures for efficiently storing and manipulating large datasets. This tutorial will cover the basic functionalities of Pandas, including creating DataFrames, indexing, cleaning, and exploring data. 


### **Creating Dataframes**

DataFrame is a two-dimensional, tabular data structure with labeled axes (rows and columns). It is one of the key data structures provided by Pandas and is widely used for data manipulation and analysis. Creating DataFrames can be done in various ways, such as from dictionaries, lists, CSV files, Excel files, and more. Here, we'll explore some common methods for creating DataFrames.

- From lists or arrays or list of dictionaries
``` py 
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22],
        'City': ['New York', 'San Francisco', 'Los Angeles']}
# or
data = [{'Name': 'Alice', 'Age': 25, 'City': 'New York'}, 
		{'Name': 'Bob', 'Age': 30, 'City': 'San Francisco'}, 
		{'Name': 'Charlie', 'Age': 22, 'City': 'Los Angeles'}]

df = pd.DataFrame(data)
print(df)
```

This would result in a dataframe like this 

``` markdown 
      Name  Age           City
0    Alice   25       New York
1      Bob   30  San Francisco
2  Charlie   22    Los Angeles
```

- From csvs
``` py
df = pd.read_csv('data.csv')
```
- From excel 
``` py
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```
### **Exploring data**

Exploring data is a crucial step in any data analysis process, and Pandas provides a variety of functions to help you understand and analyze your dataset. Here are some common techniques for exploring data using Pandas in Python

- Basic
``` py 
import pandas as pd

# Assuming df is your DataFrame
print(df.head())     # Display the first few rows
print(df.tail())     # Display the last few rows
print(df.info())     # Display concise summary
print(df.describe())  # Descriptive statistics

```
- Indexing and selecting information
``` py 
# Selecting a single column
print(df['Column'])

# Selecting multiple columns
print(df[['Column1', 'Column2']])

# Selecting a row by label
print(df.loc[0])

# Selecting a row by integer index
print(df.iloc[0])

# Selecting rows based on a condition
print(df[df['Column'] > 25])

```
- Missing values or duplicates
``` py 
# Checking for missing values
print(df.isnull())

# Dropping rows with any missing values
df_cleaned = df.dropna()

# Filling missing values with a specific value
df_filled = df.fillna(value)

# Checking for duplicate rows
print(df.duplicated())

# Removing duplicate rows
df_no_duplicates = df.drop_duplicates()

```
- creating new columns 
``` py 
# Creating a new column based on existing columns
df['NewColumn'] = df['Column1'] + df['Column2']

```
- Handling Categorical Data
``` py 
# Converting a column to categorical
df['CategoryColumn'] = df['CategoryColumn'].astype('category')

# Getting counts of each category
print(df['CategoryColumn'].value_counts())

```
- Grouping
``` py 
# Grouping by a column and applying aggregation functions
grouped_df = df.groupby('CategoryColumn').agg({'NumericColumn': ['mean', 'sum']})
print(grouped_df)
```
- Visualization 
``` py 
import matplotlib.pyplot as plt
import seaborn as sns

# Basic plotting
df.plot(x='Column1', y='Column2', kind='scatter')
plt.show()

# Using Seaborn for more advanced plots
sns.boxplot(x='CategoryColumn', y='NumericColumn', data=df)
plt.show()

```
### **Data Cleaning**

Data cleaning is an essential step in the data preparation process, ensuring that the dataset is accurate, consistent, and suitable for analysis. Pandas provides various functions to facilitate data cleaning tasks. Here are some common techniques for data cleaning using Pandas in Python

- Handling outliers 
``` py 
import numpy as np

Q1 = np.percentile(df['NumericColumn'], 25)
Q3 = np.percentile(df['NumericColumn'], 75)
IQR = Q3 - Q1

# Filtering data within 1.5 times the interquartile range
df_no_outliers = df[(df['NumericColumn'] >= Q1 - 1.5 * IQR) & (df['NumericColumn'] <= Q3 + 1.5 * IQR)]

```
-  Correcting Data Types
``` py 
df['NumericColumn'] = pd.to_numeric(df['NumericColumn'])  # Convert column to numeric
df['DateColumn'] = pd.to_datetime(df['DateColumn'])      # Convert column to datetime
df['CategoryColumn'] = df['CategoryColumn'].astype('category')  # Convert column to category

```
- Text data cleaning 
``` py 
df['TextColumn'] = df['TextColumn'].str.lower()      # Convert text to lowercase
df['TextColumn'] = df['TextColumn'].str.strip()      # Remove leading and trailing whitespaces
df['TextColumn'] = df['TextColumn'].str.replace('[^a-zA-Z ]', '')  # Remove non-alphabetic characters

```
- Renaming columns
``` py
df.rename(columns={'OldName': 'NewName'}, inplace=True)  # Rename a specific column
df.columns = ['NewColumn1', 'NewColumn2']  # Rename all columns

```
- Binarization 
``` py 
bins = [0, 25, 50, 75, 100]
labels = ['0-25', '26-50', '51-75', '76-100']

df['BinnedColumn'] = pd.cut(df['NumericColumn'], bins=bins, labels=labels)

```

- Dealing with date and time 
``` py 
df['DateColumn'] = pd.to_datetime(df['DateColumn'])  # Convert column to datetime
df['DayOfWeek'] = df['DateColumn'].dt.day_name()  # Extract day of the week
df['Month'] = df['DateColumn'].dt.month  # Extract month
```

### **Data Manipulation**

Data modification  involves making changes to the existing data, such as adding or removing columns, updating values, and creating new features. Here are some common data modification tasks using Pandas

- Adding or update an existing column
``` py
import pandas as pd

# Assuming df is your DataFrame
df['NewColumn'] = [1, 2, 3, 4, 5]

df['ExistingColumn'] = df['ExistingColumn'] * 2

```
- Drop a column in place 
``` py
# Removing a single column
df.drop('ColumnToRemove', axis=1, inplace=True)

# Removing multiple columns
columns_to_remove = ['Column1', 'Column2']
df.drop(columns=columns_to_remove, inplace=True)

```
- Applying functions 
``` py
# Applying a function to each element of a column
df['NumericColumn'] = df['NumericColumn'].apply(lambda x: x * 2)

# Applying a function to each row
df['NewColumn'] = df.apply(lambda row: row['Column1'] + row['Column2'], axis=1)

```
- Map and Reduce 
``` py
# Creating a new column based on a mapping
gender_mapping = {'M': 'Male', 'F': 'Female'}
df['Gender'] = df['Code'].map(gender_mapping)

```
- Change data type 
``` py
# Converting a column to numeric
df['NumericColumn'] = pd.to_numeric(df['NumericColumn'])

# Converting a column to datetime
df['DateColumn'] = pd.to_datetime(df['DateColumn'])

# Converting a column to category
df['CategoryColumn'] = df['CategoryColumn'].astype('category')

```
- Sorting 
``` py
# Sorting based on one or more columns
df.sort_values(by=['Column1', 'Column2'], ascending=[True, False], inplace=True)

```
- Combining data frames 
``` py
# Concatenating DataFrames along rows or columns
df_concatenated = pd.concat([df1, df2], axis=0)

```

### **Conclusion**

Pandas is an invaluable tool for anyone working with tabular data in Python. It provides a flexible and expressive framework for data manipulation, making it easier to clean, analyze, and visualize datasets. By mastering Pandas, you empower yourself to tackle a wide range of data-related tasks efficiently.

### References 

1. [Pandas documentation](https://pandas.pydata.org/docs/user_guide/index.html)