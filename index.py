# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visuals
sns.set_style("whitegrid")

# Load the Iris dataset
try:
    # Using sklearn's load_iris to get the dataset
    iris = load_iris()
    # Convert to pandas DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check dataset structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Since Iris dataset is clean, no further cleaning is needed
# If there were missing values, we could handle them like this:
# df.fillna(df.mean(), inplace=True)  # Fill with mean
# df.dropna(inplace=True)  # Drop rows with missing values

# Observations
print("\nObservations: The Iris dataset contains 150 rows and 5 columns (4 numerical features and 1 categorical species column). There are no missing values.")

# Compute basic statistics for numerical columns
print("\nBasic Statistics for Numerical Columns:")
print(df.describe())

# Group by species and compute mean for numerical columns
print("\nMean of Numerical Columns by Species:")
group_means = df.groupby('species').mean()
print(group_means)

# Findings
print("\nFindings:")
print("- The dataset has three species: setosa, versicolor, and virginica.")
print("- Setosa has the smallest average sepal width but the largest average petal length.")
print("- Virginica has the largest average sepal length and petal width.")
print("- Versicolor lies between the other two species in most measurements.")

# Create a figure with subplots
plt.figure(figsize=(15, 10))

# 1. Line Chart: Mean feature values over species (not truly time-series, but shows trend across categories)
plt.subplot(2, 2, 1)
for column in df.columns[:-1]:  # Exclude species
    plt.plot(group_means.index, group_means[column], marker='o', label=column)
plt.title('Mean Feature Values Across Species')
plt.xlabel('Species')
plt.ylabel('Mean Value (cm)')
plt.legend()
plt.xticks(rotation=45)

# 2. Bar Chart: Average petal length per species
plt.subplot(2, 2, 2)
group_means['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.xticks(rotation=45)

# 3. Histogram: Distribution of sepal length
plt.subplot(2, 2, 3)
plt.hist(df['sepal length (cm)'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')

# 4. Scatter Plot: Sepal length vs. petal length
plt.subplot(2, 2, 4)
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'], 
                label=species, alpha=0.6)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()

# Additional Observations from Visualizations
print("\nVisualization Insights:")
print("- The line chart shows distinct trends in feature means across species, with petal length varying the most.")
print("- The bar chart highlights that virginica has the longest average petal length, followed by versicolor and setosa.")
print("- The histogram shows that sepal length is roughly normally distributed with a peak around 5-6 cm.")
print("- The scatter plot reveals clear separation between species, especially for setosa, indicating strong correlation between sepal and petal length for classification.")

print("\nConclusion:")
print("The Iris dataset analysis reveals distinct characteristics among the three species. Setosa is easily separable based on its smaller petal measurements, while versicolor and virginica overlap more but can be distinguished by petal length and width. The visualizations effectively highlight these differences, making the dataset suitable for classification tasks.")