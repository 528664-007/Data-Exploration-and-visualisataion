import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid')

# -------------------------------
# 🔹 Part A: Temperature Analysis
# -------------------------------
# Sample data creation
temp_data = {
    'City': ['CityA'] * 12 + ['CityB'] * 12,
    'Date': pd.date_range(start='2024-01-01', periods=12, freq='M').tolist() * 2,
    'Temperature': [10, 12, 15, 18, 22, 30, 35, 33, 25, 20, 15, 12] + 
                   [8, 9, 13, 17, 21, 29, 34, 32, 24, 19, 14, 11]
}

df = pd.DataFrame(temp_data)
df['Month'] = df['Date'].dt.strftime('%B')
df['Month_Num'] = df['Date'].dt.month

# Group by City and Month
monthly_temp = df.groupby(['City', 'Month_Num', 'Month'])['Temperature'].sum().reset_index()

# 🔄 Pivot for summary table
pivot_temp = monthly_temp.pivot(index='City', columns='Month', values='Temperature')
print("📊 Monthly Temperature Summary:\n", pivot_temp)

# 🌞 Identify hottest city in summer
summer_months = ['June', 'July', 'August']
summer_totals = monthly_temp[monthly_temp['Month'].isin(summer_months)].groupby('City')['Temperature'].sum()
hottest_city = summer_totals.idxmax()
print(f"\n🔥 City with highest summer temperature total: {hottest_city}")

# -------------------------------
# 📈 Visualizations for Temperature
# -------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_temp, x='Month', y='Temperature', hue='City',
            order=pd.date_range('2024-01-01', periods=12, freq='M').strftime('%B'))
plt.title("Bar Plot - Monthly Temperature per City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_temp, x='Month_Num', y='Temperature', hue='City', marker='o')
plt.title("Line Plot - Temperature Trend per City")
plt.xticks(monthly_temp['Month_Num'].unique(), labels=monthly_temp['Month'].unique(), rotation=45)
plt.xlabel("Month")
plt.ylabel("Temperature")
plt.tight_layout()
plt.show()

# Heatmap (pivot again for heatmap)
temp_heatmap = monthly_temp.pivot(index='City', columns='Month', values='Temperature')
plt.figure(figsize=(10, 5))
sns.heatmap(temp_heatmap, annot=True, cmap='YlOrRd')
plt.title("Heatmap - City-wise Monthly Temperatures")
plt.tight_layout()
plt.show()

# -------------------------------
# 🔹 Part B: Work Hours Analysis
# -------------------------------
emp_data = {
    'Employee': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
    'Department': ['HR', 'IT', 'IT', 'Sales', 'HR', 'Sales'],
    'Work_Hours': [160, 175, 180, 170, 150, 165]
}
emp_df = pd.DataFrame(emp_data)

# Grouping
dept_summary = emp_df.groupby('Department').agg(
    Total_Hours=('Work_Hours', 'sum'),
    Avg_Hours=('Work_Hours', 'mean'),
    Count=('Work_Hours', 'count')
).reset_index()

print("\n🧾 Department Summary:\n", dept_summary)

# Max average hours department
top_avg_dept = dept_summary.loc[dept_summary['Avg_Hours'].idxmax(), 'Department']
print(f"\n🏆 Department with highest average hours: {top_avg_dept}")

# Pivot (demo use)
pivot_hours = emp_df.pivot_table(index='Department', values='Work_Hours', aggfunc=['sum', 'mean'])
print("\n📌 Pivot Table of Department Work Hours:\n", pivot_hours)

# -------------------------------
# 📉 Visualizations for Work Hours
# -------------------------------

# Bar plot of average hours
plt.figure(figsize=(8, 5))
sns.barplot(data=dept_summary, x='Department', y='Avg_Hours', palette='Set2')
plt.title("Bar Plot - Average Work Hours by Department")
plt.tight_layout()
plt.show()

# Box plot to show distribution
plt.figure(figsize=(8, 5))
sns.boxplot(data=emp_df, x='Department', y='Work_Hours', palette='Set3')
plt.title("Box Plot - Work Hour Distribution")
plt.tight_layout()
plt.show()

# Swarm plot (alternative view)
plt.figure(figsize=(8, 5))
sns.swarmplot(data=emp_df, x='Department', y='Work_Hours', hue='Employee', palette='Dark2', size=8)
plt.title("Swarm Plot - Work Hours by Employee & Department")
plt.tight_layout()
plt.show()
