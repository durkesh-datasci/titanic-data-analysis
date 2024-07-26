import pandas as pd
#I have done this in google colab, make sure you give your file path correctly.
# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Display the first few rows of the dataset
df.head()
# Check for missing values
df.isnull().sum()
# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Cabin with 'Unknown'
df['Cabin'].fillna('Unknown', inplace=True)

# Fill missing Embarked with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Check again for missing values
df.isnull().sum()
# Drop irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
# Convert Sex to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numerical
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Convert Cabin to numerical by creating a feature indicating whether a cabin was assigned or not
df['Cabin'] = df['Cabin'].apply(lambda x: 0 if x == 'Unknown' else 1)
# Summary statistics
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt

# Set the plot style
plt.style.use('dark_background')

# Customize seaborn's default color palette
sns.set_palette('dark')

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', annot_kws={'color': 'white'}, cbar_kws={'label': 'Correlation'}, linecolor='black', linewidths=1)
plt.title('Correlation Matrix', color='white')
plt.show()

# Visualize the distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30, color='white', edgecolor='black')
plt.title('Age Distribution', color='white')
plt.xlabel('Age', color='white')
plt.ylabel('Count', color='white')
plt.show()

# Visualize survival rate by Sex
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=df, palette=['white'])
plt.title('Survival Rate by Sex', color='white')
plt.xlabel('Sex', color='white')
plt.ylabel('Survival Rate', color='white')
plt.show()

# Visualize survival rate by Pclass
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=df, palette=['white'])
plt.title('Survival Rate by Pclass', color='white')
plt.xlabel('Pclass', color='white')
plt.ylabel('Survival Rate', color='white')
plt.show()
# Survival rate by Age
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Survived'] == 1]['Age'], kde=True, bins=30, color='white', label='Survived', edgecolor='black')
sns.histplot(df[df['Survived'] == 0]['Age'], kde=True, bins=30, color='grey', label='Not Survived', edgecolor='black')
plt.title('Survival Rate by Age', color='white')
plt.xlabel('Age', color='white')
plt.ylabel('Count', color='white')
plt.legend()
plt.show()

# Survival rate by Fare
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Survived'] == 1]['Fare'], kde=True, bins=30, color='white', label='Survived', edgecolor='black')
sns.histplot(df[df['Survived'] == 0]['Fare'], kde=True, bins=30, color='grey', label='Not Survived', edgecolor='black')
plt.title('Survival Rate by Fare', color='white')
plt.xlabel('Fare', color='white')
plt.ylabel('Count', color='white')
plt.legend()
plt.show()
