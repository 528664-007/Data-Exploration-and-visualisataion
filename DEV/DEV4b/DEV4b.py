import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Sample synthetic dataset
employee_data = {
    'Employee': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
    'Department': ['HR', 'IT', 'IT', 'Sales', 'HR', 'Sales'],
    'Work_Hours': [160, 175, 180, 170, 150, 165]
}
emp_df = pd.DataFrame(employee_data)

# Group by Department
dept_summary = emp_df.groupby('Department').agg(
    Total_Hours=('Work_Hours', 'sum'),
    Average_Hours=('Work_Hours', 'mean'),
    Employee_Count=('Work_Hours', 'count')
).reset_index()

print("\nDepartment Summary Report:")
print(dept_summary)

# Highlight department with highest average hours
max_avg_dept = dept_summary.loc[dept_summary['Average_Hours'].idxmax(), 'Department']
print(f"\nDepartment with highest average working hours: {max_avg_dept}")

# Pivot (though here it's flat, for demo)
pivot_hours = emp_df.pivot_table(index='Department', values='Work_Hours', aggfunc=['sum', 'mean'])
print("\nPivot Table of Department Hours:")
print(pivot_hours)

# 4c: Visualization
plt.figure(figsize=(8, 5))
sns.barplot(data=dept_summary, x='Department', y='Average_Hours', palette='coolwarm')
plt.title('Average Work Hours by Department')
plt.tight_layout()
plt.show()
