from collections import defaultdict
import pandas as pd

# Read In Observations
data = []
file = "data/data.txt"
with open(file, 'r') as f:
    data = f.read().strip().split(',')


# Gather Counts
counts = defaultdict(lambda: defaultdict(int)) 
for i in range(len(data) - 1):
    counts[data[i]][data[i+1]] += 1

# Initialize States and Transition Matrix
states = ['S', 'R', 'C']
transition_matrix = []

# Normalize Each Row
for s in states:
    total = sum(counts[s].values())
    row = [counts[s][next_s] / total for next_s in states]
    transition_matrix.append(row)

s_count = data.count('S')
r_count = data.count('R')
c_count = data.count('C')
n = len(data)

display = pd.DataFrame({
    'Total': [n - 1, '100%'],
    'Sunny': [s_count, f'{s_count / n * 100:.1f}%'],
    'Rainy': [r_count, f'{r_count / n * 100:.1f}%'],
    'Cloudy': [c_count, f'{c_count / n * 100:.1f}%']
}, index=['Days', 'Percentage'])

print(display)

print(f'\n========== TRANSITION MATRIX (MODEL) ===========')
for row in transition_matrix:
    print(f'{row}\n')

print(f'\n========== FULL DATA ===========')
print(data)