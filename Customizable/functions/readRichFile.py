import os
import pandas as pd
def readRichFile(path, scoreThresh):
    # le fichiers existe-il?
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} non trovato.")
    df = pd.read_csv(path)
    # Filter rows with score above threshold and valid track_id
    df_filter = df[df['score'] >= scoreThresh]
    df_filter = df_filter[df_filter["track_id"].fillna(-1).astype(int) >= 0]

    # Remove duplicates: for each (track_id, frame), keep the detection with highest score
    df_filter = (df_filter.sort_values(["track_id", "frame", "score"], ascending=[True, True, False])
            .drop_duplicates(["track_id", "frame"], keep="first"))
    # Conversion des frames: frame1 → frame0 (indexation à partir de 0)
    df_filter["frame0"] = df_filter["frame"].astype(int) - 1
    return df_filter
