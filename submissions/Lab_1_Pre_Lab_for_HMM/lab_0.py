import numpy as np
import pandas as pd
import random

states = ['Sunny', 'Rainy']

transition_matrix = np.array([
    [0.9, 0.1], # Sunny, Rainy
    [0.5, 0.5] # Rainy, Sunny
])

def weather_markov_chain(n_predictions=1000):
    predictions = []
    current_state = random.randint(0, 1)
    predictions.append(states[current_state])

    for _ in range(n_predictions-1):
        current_state = np.random.choice(
            [0, 1], p=transition_matrix[current_state]
        )
        predictions.append(states[current_state])

    return predictions

if __name__ == "__main__":
    # Gather Predictions
    preds = weather_markov_chain()

    # Table 1
    n = len(preds)
    sunny, rainy = preds.count('Sunny'), preds.count('Rainy')

    table_1 = pd.DataFrame({
        'Total': [n, '100%'],
        'Sunny': [sunny, f'{sunny / n * 100:.1f}%'],
        'Rainy': [rainy, f'{rainy / n * 100:.1f}%'],
    }, index=['Days', 'Percentage'])

    # Table 2
    after_sunny = [preds[i+1] for i in range(n-1) if preds[i] == 'Sunny']
    as_total = len(after_sunny)
    as_sunny = after_sunny.count('Sunny')
    as_rainy = after_sunny.count('Rainy')
    table_2 = pd.DataFrame({
        'Total': [as_total, '100%'],
        'Sunny': [as_sunny, f'{as_sunny / as_total * 100:.1f}%'],
        'Rainy': [as_rainy, f'{as_rainy / as_total * 100:.1f}%']
    }, index=['Days', 'Percentage'])

    # Table 3
    before_sunny = [preds[i-1] for i in range(1, n) if preds[i] == 'Sunny']
    bs_total = len(before_sunny)
    bs_sunny = before_sunny.count('Sunny')
    bs_rainy = before_sunny.count('Rainy')
    table_3 = pd.DataFrame({
        'Total': [bs_total, '100%'],
        'Sunny': [bs_sunny, f'{bs_sunny / bs_total * 100:.1f}%'],
        'Rainy': [bs_rainy, f'{bs_rainy / bs_total * 100:.1f}%']
    }, index=['Days', 'Percentage'])


    # Display Results
    print(f"\n======= TABLE 1 =======")
    print(table_1)

    print(f"\n======= TABLE 2 =======")
    print(table_2)

    print(f"\n======= TABLE 3 =======")
    print(table_3)
    