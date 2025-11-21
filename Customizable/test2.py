import pandas as pd

df = pd.DataFrame({
    'A': [1, 'x', 3, 'x', 5],
    'B': ['x', 2, 'x', 4, 5],
    'C': [10, 20, 30, 40, 50]
})

# Remplacer seulement dans certaines colonnes
df_specific = df.replace({'A': {'x': 'remplacé_A'}, 'B': {'x': 'remplacé_B'}})
print("Remplacement par colonne:")
print(df_specific)

# Utiliser where pour des conditions complexes
df_where = df.where(df != 'x', 'conditionnel')
print("\nAvec where:")
print(df_where)