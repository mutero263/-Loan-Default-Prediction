import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"
df = pd.read_csv(file_path)

# Check if 'default' column exists
if 'default' not in df.columns:
    raise ValueError("Column 'default' not found in the dataset.")

# Count loan statuses
# 0 = Good (no default), 1 = Bad (loan defaulted)
default_counts = df['default'].value_counts()

# Map to readable labels
labels = ['Good', 'Bad']
sizes = [default_counts.get(0, 0), default_counts.get(1, 0)]
total_count = sum(sizes)

# Define colors
colors = ['#00008B', '#C71585']  

#  Create donut chart
fig, ax = plt.subplots(figsize=(7, 7))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct=lambda pct: f'{pct:.1f}%\n({int(pct / 100 * total_count)})',
    startangle=90,
    colors=colors,
    textprops={'color': "white", 'fontsize': 12, 'weight': 'bold'},
    pctdistance=0.85
)

# Add donut hole (white circle in center)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax.add_artist(centre_circle)

# Add total in the center
plt.text(
    0, 0,
    f'Total\n{total_count}',
    ha='center', va='center',
    fontsize=20,
    weight='bold'
)

# Equal aspect ratio ensures the pie is circular
ax.axis('equal')

#  Add title
plt.title('Loan Status Distribution', fontsize=16, pad=20, weight='bold')

# Display counts in console
print("Loan Status Count:")
print(f"Good (default=0): {sizes[0]}")
print(f"Bad  (default=1): {sizes[1]}")
print(f"Total Loans: {total_count}")

#  Show the plot
plt.tight_layout()
plt.show()