import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# display 5 rows(default behaviour of head())
data = pd.read_csv("ecommerce_customer_data.csv")
print(data.head())
print("\n")

# Summary statistics for numeric columns
numeric_summary = data.describe()
print(numeric_summary)
print("\n")

# Summary for non-numeric columns
categorical_summary = data.describe(include='object')
print(categorical_summary) 
print("\n") 

# Calculate churn rate 
data['Churned'] = data['Total_Purchases'] == 0
churn_rate = data['Churned'].mean()
print(f"Churn Rate:{churn_rate}")

# Histogram for 'Age'
plt.figure(figsize=(10, 6))
plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Bar chart for 'Gender'
gender_counts = data['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']
plt.bar(gender_counts['Gender'], gender_counts['Count'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 'Product_Browsing_Time' vs 'Total_Pages_Viewed'
plt.scatter(data['Product_Browsing_Time'], data['Total_Pages_Viewed'])
# Linear regression
m, b = np.polyfit(data['Product_Browsing_Time'], data['Total_Pages_Viewed'], 1)
plt.plot(data['Product_Browsing_Time'], m * data['Product_Browsing_Time'] + b, color='red')

plt.title('Product Browsing Time vs. Total Pages Viewed')
plt.xlabel('Product Browsing Time')
plt.ylabel('Total Pages Viewed')
plt.show()

# # Grouped Analysis
# Average Total Pages Viewed by Gender
gender_grouped = data.groupby('Gender')['Total_Pages_Viewed'].mean().reset_index()
plt.bar(gender_grouped['Gender'],gender_grouped['Total_Pages_Viewed'],color=['blue', 'orange'])
plt.title('Average Total Pages Viewed by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Total Pages Viewed')

plt.show()

# Average Total Pages Viewed by Devices
devices_grouped = data.groupby('Device_Type')['Total_Pages_Viewed'].mean().reset_index()
plt.bar(devices_grouped['Device_Type'], devices_grouped['Total_Pages_Viewed'], color=['green', 'red', 'purple'])

plt.title('Average Total Pages Viewed by Devices')
plt.xlabel('Device Type')
plt.ylabel('Average Total Pages Viewed')
plt.show()

data['CLV'] = (data['Total_Purchases'] * data['Total_Pages_Viewed']) / data['Age']
data['Segment'] = pd.cut(data['CLV'], bins=[1, 2.5, 5, float('inf')],
                         labels=['Low Value', 'Medium Value', 'High Value'])

# Count the occurrences of each segment
segment_counts = data['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']
plt.bar(segment_counts['Segment'], segment_counts['Count'], color=['blue', 'orange', 'green'])
plt.title('Customer Segmentation by CLV')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.show()

# Funnel analysis
funnel_data = data[['Product_Browsing_Time', 'Items_Added_to_Cart', 'Total_Purchases']]
funnel_data = funnel_data.groupby(['Product_Browsing_Time', 'Items_Added_to_Cart']).sum().reset_index()

# Sort data by 'Product_Browsing_Time' for proper funnel visualization
funnel_data.sort_values(by='Product_Browsing_Time', inplace=True)

fig, ax = plt.subplots(figsize=(8, 6))

# Draw lines
for i in range(len(funnel_data) - 1):
    ax.plot(
        [i, i + 1],
        [funnel_data['Items_Added_to_Cart'].iloc[i], funnel_data['Items_Added_to_Cart'].iloc[i + 1]],
        marker='o',
        label=funnel_data['Total_Purchases'].iloc[i],
    )

plt.title('Conversion Funnel')
plt.xlabel('Product Browsing Time')
plt.ylabel('Items Added to Cart')
plt.show()
