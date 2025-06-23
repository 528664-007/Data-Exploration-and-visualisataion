import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample synthetic dataset
data = {
    'City': ['CityA'] * 12 + ['CityB'] * 12,
    'Date': pd.date_range(start='2024-01-01', periods=12, freq='M').tolist() * 2,
    'Temperature': [10, 12, 15, 18, 22, 30, 35, 33, 25, 20, 15, 12] + 
                   [8, 9, 13, 17, 21, 29, 34, 32, 24, 19, 14, 11]
}

df = pd.DataFrame(data)
df['Month'] = df['Date'].dt.month_name()
df['Month_Num'] = df['Date'].dt.month

# Group by City and Month, sum temperatures
monthly_temp = df.groupby(['City', 'Month_Num', 'Month'])['Temperature'].sum().reset_index()

# Pivot to get month-wise summary
pivot_temp = monthly_temp.pivot(index='City', columns='Month', values='Temperature')
print("Monthly Temperature Summary:")
print(pivot_temp)

# Identify city with highest total temperature in summer (June, July, August)
summer_months = ['June', 'July', 'August']
monthly_temp['Is_Summer'] = monthly_temp['Month'].isin(summer_months)
summer_totals = monthly_temp[monthly_temp['Is_Summer']].groupby('City')['Temperature'].sum()
hottest_city = summer_totals.idxmax()

print(f"\nCity with highest summer total temperature: {hottest_city}")

# 4c: Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_temp, x='Month', y='Temperature', hue='City', order=pd.date_range('2024-01-01', periods=12, freq='M').strftime('%B').unique())
plt.title('Monthly Temperature per City')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
