import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats
import seaborn as sns

st.title("Fraud Indicators in Medicare 2021")

df = pd.read_csv('Medicare_Benford.csv')


df['Avg_Sbmtd_Chrg_str'] = df['Avg_Sbmtd_Chrg'].astype(str)

df['Avg_Sbmtd_Chrg_str'] = df['Avg_Sbmtd_Chrg_str'].str.lstrip('0').str.replace('.', '').str.lstrip('0')

df['first_digit'] = df['Avg_Sbmtd_Chrg_str'].str.get(0)
df['second_digit'] = df['Avg_Sbmtd_Chrg_str'].str.get(1)
df['first_second_digit'] = df['first_digit']+df['second_digit']
df_filter = df[df['Avg_Sbmtd_Chrg']>=10]

observed_first = np.bincount(df.first_digit)[1:]
observed_first = observed_first / len(df.first_digit)
expected_first = np.log10(1 + 1 / np.arange(1, 10))

title = 'First-Order First Digit Test'

plt.figure(figsize=(16, 9))
plt.bar(np.arange(1, 10), observed_first*100, label="Actual Distribution", color='orange', alpha=0.7)
plt.plot(np.arange(1, 10), expected_first*100, label='Expected', color='r', marker='o')
plt.xlabel(title)
plt.ylabel('Probability (%)')
plt.title(f'Benford\'s Law {title}')
plt.xticks(np.arange(1,10))
plt.legend(prop={'size': 16})
plt.show()

mad = np.mean(np.abs(observed_first - expected_first))
    
if mad <= 0.0012:
    print(f"MAD: {mad:.4f} - Close conformity")
elif mad <= 0.0018:
    print(f"MAD: {mad:.4f} - Acceptable conformity")
elif mad <= 0.0022:
    print(f"MAD: {mad:.4f} - Marginally acceptable conformity")
else:
    print(f"MAD: {mad:.4f} - Nonconformity")
    

st.pyplot(plt)
