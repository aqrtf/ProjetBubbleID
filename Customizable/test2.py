import pandas as pd

def count_ints_and_nones(lst):
    # indices des entiers
    int_indices = [i for i, x in enumerate(lst) if isinstance(x, int)]
    if not int_indices:
        return 0, 0  # aucun entier
    
    start, end = int_indices[0], int_indices[-1]
    sublist = lst[start:end+1]
    
    num_ints = sum(isinstance(x, int) for x in sublist)
    num_nones = sum(x is None for x in sublist)
    
    return num_ints, num_nones

# Exemple
lst = [None, None, 5, None, 7, None, 9, None]
print(count_ints_and_nones(lst))  # (3, 2)

