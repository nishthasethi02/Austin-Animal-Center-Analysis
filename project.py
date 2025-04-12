import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter

# Import the dataset
df = pd.read_csv("C:\\Users\\Nishtha Sethi\\Downloads\\Austin_Animal_Center_Outcomes.csv")

# Getting file information
print("First few rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Cleaning up the data
df.replace('', pd.NA, inplace=True)
df['Outcome Type'] = df['Outcome Type'].replace('Unknown', 'Other')
df['Sex upon Outcome'] = df['Sex upon Outcome'].replace('Unknown', 'Intact')
df['Animal Type'] = df['Animal Type'].replace('Unknown', 'Other')
df['Breed'] = df['Breed'].replace('Unknown', 'Mixed')
df['Color'] = df['Color'].replace('Unknown', 'Brown/Black')
df['Name'] = df['Name'].replace(np.nan, 'No Name')

for col in ['Outcome Type', 'Sex upon Outcome', 'Animal Type', 'Breed', 'Color', 'Age upon Outcome']:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])
df.drop_duplicates(inplace=True)
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
df = df.dropna(subset=['DateTime'])
df.dropna(inplace=True)
print("\nMissing values after final cleaning:")
print(df.isnull().sum())

# ========== Objective 1: Quantify "No-Kill" policy ==========
live_outcomes = ['Adoption', 'Transfer', 'Return to Owner']
df['Is_Live'] = df['Outcome Type'].isin(live_outcomes)
live_rate = df['Is_Live'].mean() * 100
non_live_rate = 100 - live_rate
print(f"\nLive Outcome Rate: {live_rate:.2f}%")
print(f"Non-Live Outcome Rate: {non_live_rate:.2f}%")
outcome_counts = df['Is_Live'].value_counts()
labels = ['Live Outcomes', 'Non-Live Outcomes']
colors = ['green', 'red']
plt.figure(figsize=(6, 6))
plt.pie(outcome_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Live vs Non-Live Outcomes')
plt.axis('equal')
plt.show()

# ========== Objective 2: Analyze outcome types across animal types ==========
grouped = df.groupby(['Animal Type', 'Outcome Type']).size().reset_index(name='Count')
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped, x='Outcome Type', y='Count', hue='Animal Type')
plt.xticks(rotation=45)
plt.title('Outcome Types by Animal Type')
plt.xlabel('Outcome Type')
plt.ylabel('Count')
plt.legend(title='Animal Type')
plt.tight_layout()
plt.show()

# ========== Objective 3: Effect of Age on Outcomes ==========
def convert_age_to_weeks(age_str):
    try:
        number, unit = age_str.split()[:2]
        number = int(number)
        unit = unit.lower()
        if 'day' in unit:
            return number / 7
        elif 'week' in unit:
            return number
        elif 'month' in unit:
            return number * 4
        elif 'year' in unit:
            return number * 52
        else:
            return None
    except:
        return None

df['AgeWeeks'] = df['Age upon Outcome'].apply(convert_age_to_weeks)
adopted_df = df[df['Outcome Type'] == 'Adoption'].copy()
adopted_df.dropna(subset=['AgeWeeks'], inplace=True)
plt.figure(figsize=(10, 6))
sns.histplot(adopted_df['AgeWeeks'], bins=30, kde=True, color='orange')
plt.title('Distribution of Age (in Weeks) Among Adopted Animals', fontsize=14)
plt.xlabel('Age in Weeks')
plt.ylabel('Number of Adopted Animals')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ========== Objective 4: Effect of Neutered/Spayed on Outcome ==========
print("\nOutcome distribution by Neutering status:")
neuter_outcome = pd.crosstab(df['Sex upon Outcome'], df['Outcome Type'], normalize='index') * 100
print(neuter_outcome)
statuses = ['Neutered Male', 'Spayed Female', 'Intact Male', 'Intact Female']
df_filtered = df[df['Sex upon Outcome'].isin(statuses)]
adopt_counts = df_filtered[df_filtered['Outcome Type'] == 'Adoption']['Sex upon Outcome'].value_counts()
total_counts = df_filtered['Sex upon Outcome'].value_counts()
adoption_rates = (adopt_counts / total_counts * 100).reindex(statuses).fillna(0)
print("\nAdoption Rate (%) by Sex upon Outcome:")
print(adoption_rates)
plt.figure(figsize=(8, 5))
sns.barplot(x=adoption_rates.index, y=adoption_rates.values, 
            hue=adoption_rates.index, palette='pastel', legend=False)
plt.title('Adoption Rate by Neutered/Spayed/Intact Status')
plt.ylabel('Adoption Rate (%)')
plt.xlabel('Sex upon Outcome')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ========== Objective 5: Adoption Trends Over Time ==========
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['YearMonth'] = df['DateTime'].dt.to_period('M')
adoptions = df[df['Outcome Type'] == 'Adoption']
monthly_adoptions = adoptions.groupby('YearMonth').size().reset_index(name='AdoptionCount')
monthly_adoptions['YearMonth'] = monthly_adoptions['YearMonth'].dt.to_timestamp()
yearly_adoptions = adoptions.groupby('Year').size().reset_index(name='AdoptionCount')
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
ax1 = plt.gca()
sns.lineplot(data=monthly_adoptions, x='YearMonth', y='AdoptionCount', 
             color='#2c7bb6', linewidth=2, marker='o', markersize=5, ax=ax1)
plt.title('Monthly Adoption Trends (2013-2025)', fontsize=14, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Adoptions', fontsize=12)
ax1.xaxis.set_major_locator(YearLocator())
ax1.xaxis.set_major_formatter(DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.subplot(1, 2, 2)
sns.lineplot(data=yearly_adoptions, x='Year', y='AdoptionCount', marker='o')
plt.title('Yearly Adoption Trends (2013-2025)')
plt.xlabel('Year')
plt.ylabel('Number of Adoptions')
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== Objective 6: Color impact on outcomes ==========
adopted_df = df[df['Outcome Type'] == 'Adoption']
color_adoptions = adopted_df.groupby('Color').size().reset_index(name='Adoption Count')
top_colors = color_adoptions.sort_values(by='Adoption Count', ascending=False).head(15)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_colors, x='Adoption Count', y='Color', hue='Color', palette='viridis', legend=False)
plt.title('Top 15 Coat Colors by Number of Adoptions')
plt.xlabel('Number of Adoptions')
plt.ylabel('Coat Color')
plt.tight_layout()
plt.show()

# ========== Objective 7: Chances of specific outcomes by species ==========
species_outcome_chances = df.groupby(['Animal Type', 'Outcome Type']).size().unstack()
species_outcome_percent = species_outcome_chances.div(species_outcome_chances.sum(axis=1), axis=0) * 100
print("\nChances of Specific Outcomes by Species (in %):")
print(species_outcome_percent.round(2))
species_outcome_percent.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Chances of Specific Outcomes by Species')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.legend(title='Outcome Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ========== Objective 8: Chances of specific outcomes by breed ==========
breed_outcome_counts = df.groupby(['Breed', 'Outcome Type']).size().unstack(fill_value=0)
breed_outcome_percent = breed_outcome_counts.div(breed_outcome_counts.sum(axis=1), axis=0) * 100
top_breeds = df['Breed'].value_counts().head(10).index
top_breed_outcomes = breed_outcome_percent.loc[top_breeds]
print("\nChances of Specific Outcomes by Top Breeds (in %):")
print(top_breed_outcomes.round(2))
top_breed_outcomes.plot(kind='bar', stacked=True, colormap='tab20')
plt.title('Chances of Specific Outcomes by Top 10 Breeds')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Outcome Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ========== Correlation Heatmap using AgeWeeks ==========
plt.figure(figsize=(6, 4))
sns.heatmap(df[['AgeWeeks', 'Is_Live']].dropna().corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Age in Weeks vs Live Outcome)')
plt.show()

def convert_age_to_days(age_str):
    try:
        number, unit = age_str.split()[:2]
        number = int(number)
        unit = unit.lower()
        if 'day' in unit:
            return number
        elif 'week' in unit:
            return number * 7
        elif 'month' in unit:
            return number * 30
        elif 'year' in unit:
            return number * 365
        else:
            return None
    except:
        return None

df['AgeDays'] = df['Age upon Outcome'].apply(convert_age_to_days)

# ========== Correlation Heatmap using AgeDays ==========
plt.figure(figsize=(6, 4))
sns.heatmap(df[['AgeDays', 'Is_Live']].dropna().corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Age in Days vs Live Outcome)')
plt.show()

# ========== Boxplot to show AgeWeeks by Outcome Type ==========
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Outcome Type', y='AgeWeeks')
plt.xticks(rotation=45)
plt.title('Age (in Weeks) Distribution by Outcome Type (Boxplot)')
plt.ylabel('Age in Weeks')
plt.xlabel('Outcome Type')
plt.tight_layout()
plt.show()

# ========== Boxplot to show AgeDays by Outcome Type ==========
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Outcome Type', y='AgeDays')
plt.xticks(rotation=45)
plt.title('Age (in Days) Distribution by Outcome Type (Boxplot)')
plt.ylabel('Age in Days')
plt.xlabel('Outcome Type')
plt.tight_layout()
plt.show()
