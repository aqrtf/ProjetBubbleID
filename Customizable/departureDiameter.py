def ComputeDepartureDiameter(self,
                                *,
                                k: int = 3,
                                fit_kind: str = "linear",
                                class_idx_attached: int = 2,
                                class_idx_detached: int = 0,
                                class_idx_unknown: int = 1,
                                tolerate_unknown_gap: int = 1,
                                min_attached_run: int = 1,
                                out_csv: str | None = None):
    """
    CALCULE LE DIAMETRE DE DEPART DES BULLES
    Objectif: Determiner le diamètre des bulles au moment où elles se détachent
    """
    import os, math, csv
    import numpy as np, pandas as pd

    # =============================================================================
    # SECTION 1: INITIALISATION ET CHARGEMENT DES FICHIERS
    # =============================================================================
    
    # Chemins vers les fichiers de données
    rich_csv = os.path.join(self.savefolder, f"rich_{self.extension}.csv")          # Données enrichies des bulles
    frames_path = os.path.join(self.savefolder, f'frames_{self.extension}.npy')     # Liste des frames par bulle
    bubclass_path = os.path.join(self.savefolder, f'bubclass_{self.extension}.npy') # Classification des bulles
    
    # Fichier de sortie par défaut
    if out_csv is None:
        out_csv = os.path.join(self.savefolder, f"departure_{self.extension}.csv")
    
    # VÉRIFICATIONS DE SÉCURITÉ: les fichiers existent-ils?
    if not os.path.isfile(rich_csv):
        raise FileNotFoundError(f"{rich_csv} non trovato.")
    if not os.path.isfile(frames_path) or not os.path.isfile(bubclass_path):
        raise FileNotFoundError("frames_*.npy o bubclass_*.npy non trovati.")
    if not getattr(self, "mm_per_px", None):
        raise RuntimeError("mm_per_px non impostato.")

    # Conversion pixels → millimètres
    mm_per_px = float(self.mm_per_px)
    
    # Chargement et préparation du DataFrame principal
    df = pd.read_csv(rich_csv)
    # Nettoyage des noms de colonnes: minuscules et sans espaces
    df.columns = [c.strip().lower() for c in df.columns]
    # Suppression des colonnes dupliquées
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep='first')]
    
    # Gestion des noms de colonnes alternatifs
    if "track_id" not in df.columns and "tid" in df.columns:
        df = df.rename(columns={"tid": "track_id"})
    if "frame" not in df.columns: 
        raise ValueError("Manca 'frame' nel rich CSV.")
    
    # Conversion des frames: frame1 → frame0 (indexation à partir de 0)
    df["frame0"] = df["frame"].astype(int) - 1
    # Filtrage: seulement les tracks avec ID valide (≥ 0)
    df = df[df["track_id"].fillna(-1).astype(int) >= 0]

    # VÉRIFICATION DES COLONNES REQUISES
    need = {"area_px", "perim_px", "feret_max_px", "feret_min_px", "ell_major_px", "ell_minor_px", "score"}
    miss = [c for c in need if c not in df.columns]
    if miss: 
        raise ValueError(f"Mancano colonne nel rich CSV: {miss}")

    # Chargement des arrays numpy
    frames_arr = np.load(frames_path, allow_pickle=True)    # frames_arr[tid] = liste des frames de la bulle tid
    bubclass_arr = np.load(bubclass_path, allow_pickle=True) # bubclass_arr[tid] = liste des classifications

    # =============================================================================
    # SECTION 2: FONCTIONS AUXILIAIRES
    # =============================================================================

    def _smooth(frames_list, labels_list):
        """
        LISSE LES CLASSIFICATIONS: comble les petits gaps 'inconnus' entre états 'attachés'
        Exemple: [A, U, U, A] avec tolerate_unknown_gap=2 → [A, A, A, A]
        """
        if not frames_list: 
            return frames_list, labels_list
            
        # Tri par frame
        idx = np.argsort(frames_list)
        fr = [int(frames_list[i]) for i in idx]
        lb = [int(labels_list[i]) for i in idx]
        out = lb[:]  # Copie pour modification
        
        # CORRECTION DES GAPS INCONNUS
        if tolerate_unknown_gap > 0:
            i = 0
            n = len(out)
            while i < n:
                if out[i] == class_idx_attached:
                    j = i + 1
                    unk = 0
                    # Compter les 'inconnus' consécutifs
                    while j < n and out[j] == class_idx_unknown:
                        unk += 1
                        j += 1
                    # Si gap acceptable et on retombe sur 'attaché'
                    if 0 < unk <= tolerate_unknown_gap and j < n and out[j] == class_idx_attached:
                        # Convertir les 'inconnus' en 'attachés'
                        for k2 in range(i + 1, j): 
                            out[k2] = class_idx_attached
                        i = j
                        continue
                i += 1
        return fr, out

    def _find_departure(frames0, labels):
        """
        TROUVE LE MOMENT DE DÉTACHEMENT D'UNE BULLE
        Retourne: (attach_start, attach_end_i, last_attached, detach_frame, fr_s)
        """
        # Application du lissage
        fr_s, lb_s = _smooth(frames0, labels)
        
        attach_start = None    # Premier frame attaché
        attach_end_i = None   # Index du dernier frame attaché
        run = 0               # Compteur de frames attachés consécutifs
        
        # RECHERCHE DE LA SÉQUENCE ATTACHÉE
        for i, lab in enumerate(lb_s):
            if lab == class_idx_attached:
                run += 1
                if attach_start is None: 
                    attach_start = fr_s[i]  # Premier frame attaché
                attach_end_i = i  # Dernier frame attaché (mis à jour à chaque fois)
            else:
                # Si séquence trop courte, on reset
                if run >= min_attached_run: 
                    break  # Séquence valide trouvée
                attach_start = None
                attach_end_i = None
                run = 0
                
        # VÉRIFICATION: séquence attachée valide?
        if attach_start is None or run < min_attached_run:
            return None, None, None, None, None
            
        last_attached = fr_s[attach_end_i]  # Dernier frame attaché
        detach_frame = None
        
        # RECHERCHE DU PREMIER FRAME DÉTACHÉ APRÈS LA SÉQUENCE ATTACHÉE
        for j in range(attach_end_i + 1, len(lb_s)):
            if lb_s[j] == class_idx_detached:
                detach_frame = fr_s[j]  # Premier frame détaché
                break
                
        return attach_start, attach_end_i, last_attached, detach_frame, fr_s

    def _series_for_track(tid, col_name):
        """
        RÉCUPÈRE LA SÉRIE TEMPORELLE D'UNE PROPRIÉTÉ POUR UNE BULLE DONNÉE
        Exemple: série des aires (area_px) pour la bulle tid
        """
        # Sécurité: suppression des doublons de colonnes
        dfl = df.loc[:, ~pd.Index(df.columns).duplicated(keep='first')]

        # Colonnes à extraire
        cols = ["frame0", col_name]
        # Utiliser le score comme tie-break si disponible (sauf pour la colonne score elle-même)
        use_tiebreak = (col_name != "score") and ("score" in dfl.columns)
        if use_tiebreak:
            cols.append("score")

        # Filtrage: seulement la bulle concernée
        sdf = dfl[dfl["track_id"].astype(int) == tid][cols].copy()
        if sdf.empty:
            return []

        # GESTION DES DOUBLONS: garder l'entrée avec le meilleur score
        if use_tiebreak:
            sdf = (sdf.sort_values(["frame0", "score"], ascending=[True, False])
                    .drop_duplicates("frame0", keep="first"))
        else:
            sdf = sdf.sort_values(["frame0"]).drop_duplicates("frame0", keep="first")

        # Construction de la liste (frame, valeur)
        pairs = []
        for r in sdf.itertuples(index=False):
            f0 = int(getattr(r, "frame0"))
            v = getattr(r, col_name)
            try:
                v = float(v)  # Conversion en float
            except Exception:
                continue
            if np.isfinite(v):  # Vérification que la valeur est finie (pas NaN/inf)
                pairs.append((f0, v))
        return pairs

    def _discrete_at(series, f0):
        """
        VALEUR DISCÈTE: retourne la dernière valeur disponible au frame f0 ou avant
        Méthode conservatrice: prend la dernière valeur connue
        """
        if not series: 
            return math.nan
        series = sorted(series, key=lambda x: x[0])  # Tri par frame
        # Recherche à rebours: dernière valeur ≤ f0
        for f, v in reversed(series):
            if f == f0: 
                return v
            if f < f0:  
                return v
        return math.nan

    def _interp_at(series, f_eval, deg):
        """
        INTERPOLATION: extrapole la valeur au frame f_eval using les k dernières valeurs
        Utilise un polynôme de degré 1 (linéaire) ou 2 (quadratique)
        """
        if f_eval is None or not series: 
            return math.nan
            
        # Points avant le frame d'évaluation
        pre = [(f, v) for (f, v) in sorted(series) if f < f_eval]
        if len(pre) < (deg + 1):  # Pas assez de points pour l'interpolation
            return math.nan
            
        # Extraction des k derniers points
        xs = np.array([f for f, _ in pre[-k:]], float)
        ys = np.array([v for _, v in pre[-k:]], float)
        
        # Filtrage des valeurs non finies
        m = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[m], ys[m]
        if xs.size < (deg + 1): 
            return math.nan
            
        # INTERPOLATION POLYNOMIALE
        try:
            c = np.polyfit(xs, ys, deg=1 if fit_kind.lower().startswith("lin") else 2)
            return float(np.polyval(c, float(f_eval)))
        except:
            return math.nan

    # =============================================================================
    # SECTION 3: MÉTHODES DE CALCUL DU DIAMÈTRE
    # =============================================================================
    
    # DÉFINITION DES 7 MÉTHODES DE CALCUL DU DIAMÈTRE
    methods = [
        # Diamètre équivalent circulaire à partir de l'aire
        ("area", lambda a, p, fM, fm, eM, eN: 2.0 * np.sqrt(a / np.pi)),
        # Diamètre de Feret maximum (plus grande distance entre points du contour)
        ("feret_max", lambda a, p, fM, fm, eM, eN: fM),
        # Diamètre de Feret minimum
        ("feret_min", lambda a, p, fM, fm, eM, eN: fm),
        # Axe majeur de l'ellipse ajustée
        ("ell_maj", lambda a, p, fM, fm, eM, eN: eM),
        # Axe mineur de l'ellipse ajustée
        ("ell_min", lambda a, p, fM, fm, eM, eN: eN),
        # Diamètre à partir du périmètre (circonférence → diamètre)
        ("perim", lambda a, p, fM, fm, eM, eN: p / np.pi),
        # Méthode mixte: moyenne entre area et perim
        ("mix", lambda a, p, fM, fm, eM, eN: 0.5 * (2.0 * np.sqrt(a / np.pi) + p / np.pi)),
    ]
    
    # Degré du polynôme d'interpolation
    fit_deg = 1 if fit_kind.lower().startswith("lin") else 2

    # =============================================================================
    # SECTION 4: TRAITEMENT PRINCIPAL - BOUCLE SUR TOUTES LES BULLES
    # =============================================================================
    
    rows_out = []  # Stockage des résultats
    n_tracks = len(frames_arr)  # Nombre total de bulles
    
    for tid in range(n_tracks):
        # Chargement des données pour la bulle tid
        frames0 = list(frames_arr[tid]) if isinstance(frames_arr[tid], (list, np.ndarray)) else []
        labels = list(bubclass_arr[tid]) if isinstance(bubclass_arr[tid], (list, np.ndarray)) else []
        
        # CAS 1: BULLE SANS FRAMES → DONNÉES MANQUANTES
        if not frames0:
            base = {
                "bubble_id": tid, 
                "attach_start_frame": None, 
                "last_attached_frame": None,
                "detach_frame": None, 
                "note": "no_frames", 
                "k": k, 
                "fit_kind": fit_kind,
                "conf_dep_last_attached_pct": np.nan, 
                "conf_dep_first_detached_pct": np.nan,
                "conf_dep_mean_pct": np.nan
            }
            # Initialisation de tous les diamètres à NaN
            for m, _ in methods:
                base[f"D_{m}_px_discr"] = np.nan
                base[f"D_{m}_px_interp"] = np.nan
                base[f"D_{m}_mm_discr"] = np.nan
                base[f"D_{m}_mm_interp"] = np.nan
            rows_out.append(base)
            continue

        # RECHERCHE DU DÉTACHEMENT POUR CETTE BULLE
        attach_start, attach_end_i, last_attached, detach_frame, fr_s = _find_departure(frames0, labels)
        
        # CAS 2: AUCUNE SÉQUENCE ATTACHÉE VALIDE TROUVÉE
        if attach_start is None:
            base = {
                "bubble_id": tid, 
                "attach_start_frame": None, 
                "last_attached_frame": None,
                "detach_frame": None, 
                "note": "no_attached_run", 
                "k": k, 
                "fit_kind": fit_kind,
                "conf_dep_last_attached_pct": np.nan, 
                "conf_dep_first_detached_pct": np.nan,
                "conf_dep_mean_pct": np.nan
            }
            for m, _ in methods:
                base[f"D_{m}_px_discr"] = np.nan
                base[f"D_{m}_px_interp"] = np.nan
                base[f"D_{m}_mm_discr"] = np.nan
                base[f"D_{m}_mm_interp"] = np.nan
            rows_out.append(base)
            continue

        # CAS 3: BULLE VALIDE → CALCULS COMPLETS
        
        # CHARGEMENT DES SÉRIES TEMPORELLES POUR CETTE BULLE
        ser_area = _series_for_track(tid, "area_px")      # Série des aires
        ser_perim = _series_for_track(tid, "perim_px")    # Série des périmètres
        ser_fmax = _series_for_track(tid, "feret_max_px") # Série Feret max
        ser_fmin = _series_for_track(tid, "feret_min_px") # Série Feret min
        ser_eMaj = _series_for_track(tid, "ell_major_px") # Série ellipse majeur
        ser_eMin = _series_for_track(tid, "ell_minor_px") # Série ellipse mineur
        ser_score = _series_for_track(tid, "score")       # Série des scores de confiance

        def _series_method(name):
            """
            RENVOIE LA SÉRIE CORRESPONDANT À UNE MÉTHODE DE CALCUL
            """
            if name == "area":
                return ser_area
            elif name == "perim":
                return ser_perim
            elif name == "feret_max":
                return ser_fmax
            elif name == "feret_min":
                return ser_fmin
            elif name == "ell_maj":
                return ser_eMaj
            elif name == "ell_min":
                return ser_eMin
            elif name == "mix":
                # Méthode mixte: combine area et perim
                a_dict = dict(ser_area)   # {frame: area_value}
                p_dict = dict(ser_perim)  # {frame: perim_value}
                # Frames communs aux deux séries
                common_frames = sorted(set(a_dict) & set(p_dict))
                return [(f, 0.5 * (2.0 * np.sqrt(a_dict[f] / np.pi) + p_dict[f] / np.pi)) 
                        for f in common_frames]
            else:
                return []

        def _score_at(f0):
            """
            RÉCUPÈRE LE SCORE DE CONFIANCE À UN FRAME DONNÉ
            """
            if not ser_score: 
                return np.nan
            score_dict = dict(ser_score)
            if f0 in score_dict: 
                return float(score_dict[f0])
            # Si score pas disponible au frame exact, prend le dernier connu
            prev_scores = [score_dict[f] for f in score_dict.keys() if f <= f0]
            return float(prev_scores[-1]) if prev_scores else np.nan

        # STRUCTURE DE BASE DES RÉSULTATS
        base = {
            "bubble_id": tid,
            "attach_start_frame": int(attach_start) if attach_start is not None else None,
            "last_attached_frame": int(last_attached) if last_attached is not None else None,
            "detach_frame": int(detach_frame) if detach_frame is not None else None,
            "note": "ok" if detach_frame is not None else "no_detach_found",
            "k": k, 
            "fit_kind": fit_kind
        }

        # =============================================================================
        # CALCUL DES SCORES DE CONFIANCE
        # =============================================================================
        
        sA = _score_at(last_attached)  # Score au dernier frame attaché
        sD = _score_at(detach_frame) if detach_frame is not None else np.nan  # Score au premier frame détaché
        
        # Score moyen entre dernier attaché et premier détaché
        if np.isfinite(sA) and np.isfinite(sD):
            sM = 0.5 * (sA + sD)
        elif np.isfinite(sA):
            sM = sA
        else:
            sM = np.nan
            
        # Conversion en pourcentages
        base["conf_dep_last_attached_pct"] = (sA * 100.0) if np.isfinite(sA) else np.nan
        base["conf_dep_first_detached_pct"] = (sD * 100.0) if np.isfinite(sD) else np.nan
        base["conf_dep_mean_pct"] = (sM * 100.0) if np.isfinite(sM) else np.nan

        # =============================================================================
        # CALCUL DES DIAMÈTRES POUR CHAQUE MÉTHODE
        # =============================================================================
        
        for (mname, _) in methods:
            # CONSTRUCTION DE LA SÉRIE DES DIAMÈTRES PIXELS
            if mname in {"area", "perim", "feret_max", "feret_min", "ell_maj", "ell_min"}:
                comp_series = _series_method(mname)  # Série brute (aire, périmètre, etc.)
                d_series = []
                # CONVERSION EN DIAMÈTRE
                for f, val in comp_series:
                    if mname == "area":
                        # Aire → diamètre équivalent: D = 2√(A/π)
                        d_series.append((f, 2.0 * math.sqrt(val / math.pi)))
                    elif mname == "perim":
                        # Périmètre → diamètre: D = P/π
                        d_series.append((f, val / math.pi))
                    else:
                        # Feret et ellipse: valeur directe
                        d_series.append((f, float(val)))
            else:
                # Méthode mixte: déjà calculée dans _series_method
                d_series = _series_method("mix")

            # CALCUL DES DEUX TYPES DE DIAMÈTRES
            
            # 1. DIAMÈTRE DISCÈTE: valeur au dernier frame attaché
            d_px_discr = _discrete_at(d_series, last_attached)
            
            # 2. DIAMÈTRE INTERPOLÉ: extrapolation au frame de détachement
            target_eval = detach_frame if detach_frame is not None else last_attached
            d_px_interp = _interp_at(d_series, target_eval, fit_deg)
            
            # Fallback: si interpolation échoue, utiliser la valeur discrète
            if not np.isfinite(d_px_interp): 
                d_px_interp = d_px_discr

            # STOCKAGE DES RÉSULTATS EN PIXELS ET MILLIMÈTRES
            base[f"D_{mname}_px_discr"] = d_px_discr
            base[f"D_{mname}_px_interp"] = d_px_interp
            base[f"D_{mname}_mm_discr"] = d_px_discr * mm_per_px if np.isfinite(d_px_discr) else np.nan
            base[f"D_{mname}_mm_interp"] = d_px_interp * mm_per_px if np.isfinite(d_px_interp) else np.nan

        rows_out.append(base)

    # =============================================================================
    # SECTION 5: SAUVEGARDE DES RÉSULTATS
    # =============================================================================
    
    # DÉFINITION DES COLONNES DU CSV DE SORTIE
    cols_head = [
        "bubble_id", "attach_start_frame", "last_attached_frame", "detach_frame", 
        "note", "k", "fit_kind", "conf_dep_last_attached_pct", 
        "conf_dep_first_detached_pct", "conf_dep_mean_pct"
    ]
    
    # COLONNES DES DIAMÈTRES (7 méthodes × 4 colonnes chacune)
    mcols = []
    for m, _ in methods:
        mcols += [
            f"D_{m}_px_discr",    # Diamètre discret pixels
            f"D_{m}_px_interp",   # Diamètre interpolé pixels  
            f"D_{m}_mm_discr",    # Diamètre discret mm
            f"D_{m}_mm_interp"    # Diamètre interpolé mm
        ]
    cols = cols_head + mcols

    # ÉCRITURE DU FICHIER CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            # Sécurité: assurer que toutes les colonnes existent
            for c in cols:
                if c not in r: 
                    r[c] = ""
            w.writerow(r)

    print(f"[ComputeDepartureDiameter] salvato: {out_csv}")
    return rows_out