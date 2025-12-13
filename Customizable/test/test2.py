results = []
def f(x):
    if x>0:
        return {
            "bubble_id": x,
            "first_seen_frame": 0,
            "last_seen_frame": 0,
            "n_frames_tracked": 0,
            "missing_detection": 0,
            "mean_score_pct": 0,
            "chemin": 0,
            "mergeFrame": 0,
        }, f(x-1)
    return {
            "bubble_id": x,
            "first_seen_frame": 0,
            "last_seen_frame": 0,
            "n_frames_tracked": 0,
            "missing_detection": 0,
            "mean_score_pct": 0,
            "chemin": 0,
            "mergeFrame": 0,
        }
def flatten_bubbles(node):
    """
    Aplatit la structure imbriquée de tuples en une liste de dictionnaires.
    
    Args:
    - node: La structure imbriquée (tuple ou dict).
    
    Returns:
    - list: Liste plate des dictionnaires.
    """
    if isinstance(node, tuple):
        # Si c'est un tuple, prendre le premier élément (dict) et aplatir le reste
        return [node[0]] + flatten_bubbles(node[1])
    else:
        # Si c'est un dict (cas de base), retourner une liste avec ce dict
        return [node]   
a = f(4)
print(a)
results.extend(flatten_bubbles(a))

print(results)