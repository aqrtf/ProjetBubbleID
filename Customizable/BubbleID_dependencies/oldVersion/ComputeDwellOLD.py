def ComputeAttachDwellWithSurface(self,
                                      fps=4000,
                                      class_idx_attached=2,
                                      class_idx_detached=0,
                                      class_idx_unknown=1,
                                      tolerate_unknown_gap=1,
                                      min_attached_run=2,
                                      out_csv=None):
        import os, csv, cv2, numpy as np
        import pandas as pd

        frames_path = os.path.join(self.savefolder, f'frames_{self.extension}.npy')
        bubclass_path = os.path.join(self.savefolder, f'bubclass_{self.extension}.npy')
        rich_csv = os.path.join(self.savefolder, f"rich_{self.extension}.csv")
        if not os.path.isfile(frames_path) or not os.path.isfile(bubclass_path):
            raise FileNotFoundError("frames_*.npy o bubclass_*.npy non trovati.")
        frames_arr = np.load(frames_path, allow_pickle=True)
        bubclass_arr = np.load(bubclass_path, allow_pickle=True)

        # opzionale: score per frame da rich
        df_score = None
        if os.path.isfile(rich_csv):
            df = pd.read_csv(rich_csv)
            df.columns = [c.strip().lower() for c in df.columns]
            if "track_id" not in df.columns and "tid" in df.columns:
                df = df.rename(columns={"tid": "track_id"})
            if "score" in df.columns and "frame" in df.columns:
                df["frame0"] = df["frame"].astype(int) - 1
                # tieni, per ogni (track_id, frame0), la detection con score max
                df = (df.sort_values(["track_id", "frame0", "score"], ascending=[True, True, False])
                      .drop_duplicates(["track_id", "frame0"], keep="first"))
                df_score = df[["track_id", "frame0", "score"]].copy()

        def _replaceChangedID(rich_df, changeIDList):
            if "frame" not in rich_df.columns and "frame0" in rich_df.columns:
                rich_df["frame"] = rich_df["frame0"].astype(int) +1
                
            for frame, new_id, old_id in changeIDList:
                rich_df.loc[(rich_df['frame'] >= frame) & (rich_df['track_id'] == new_id), 'track_id'] = old_id
            return rich_df

        df_score = _replaceChangedID(df_score, self.changeIDList)


        # --- FPS ---
        if fps is None:
            fps = 1.0
            if self.videopath and os.path.isfile(self.videopath):
                cap = cv2.VideoCapture(self.videopath)
                if cap.isOpened():
                    _fps = cap.get(cv2.CAP_PROP_FPS)
                    if _fps and _fps > 0:
                        fps = float(_fps)
                cap.release()

        def _smooth(frames_list, labels_list):
            if not frames_list: return frames_list, labels_list
            idx = np.argsort(frames_list)
            fr = [int(frames_list[i]) for i in idx]
            lb = [int(labels_list[i]) for i in idx]
            out = lb[:]
            if tolerate_unknown_gap > 0:
                i = 0;
                n = len(out)
                while i < n:
                    if out[i] == class_idx_attached:
                        j = i + 1;
                        unk = 0
                        while j < n and out[j] == class_idx_unknown:
                            unk += 1;
                            j += 1
                        if 0 < unk <= tolerate_unknown_gap and j < n and out[j] == class_idx_attached:
                            for k2 in range(i + 1, j): out[k2] = class_idx_attached
                            i = j;
                            continue
                    i += 1
            return fr, out

        results = []
        for tid, (fr_list, cls_list) in enumerate(zip(frames_arr, bubclass_arr)):
            fr_list = list(fr_list) if isinstance(fr_list, (list, np.ndarray)) else []
            cls_list = list(cls_list) if isinstance(cls_list, (list, np.ndarray)) else []
            if not fr_list:
                results.append({"bubble_id": tid, "attach_start_frame": None, "detach_frame": None,
                                "dwell_frames": 0, "dwell_seconds": 0.0,
                                "n_frames_tracked": 0, "n_unknown": 0,
                                "mean_score_pct": np.nan, "note": "no_frames"})
                continue

            n_unknown = sum(1 for c in cls_list if int(c) == class_idx_unknown)
            fr_s, lb_s = _smooth(fr_list, cls_list)

            # trova la prima run attached >= min_attached_run
            attach_start = None;
            attach_end_i = None;
            run = 0;
            start_i = None
            for i, lab in enumerate(lb_s):
                if lab == class_idx_attached:
                    run += 1
                    if attach_start is None:
                        attach_start = fr_s[i];
                        start_i = i
                    attach_end_i = i
                else:
                    if run >= min_attached_run: break
                    attach_start = None;
                    attach_end_i = None;
                    start_i = None;
                    run = 0

            if attach_start is None or run < min_attached_run:
                results.append({"bubble_id": tid, "attach_start_frame": None, "detach_frame": None,
                                "dwell_frames": 0, "dwell_seconds": 0.0,
                                "n_frames_tracked": len(fr_list), "n_unknown": n_unknown,
                                "mean_score_pct": np.nan, "note": "no_attached_run"})
                continue

            detach_frame = None
            for j in range(attach_end_i + 1, len(lb_s)):
                if lb_s[j] == class_idx_detached:
                    detach_frame = fr_s[j];
                    break

            if detach_frame is None:
                dwell_frames = fr_s[attach_end_i] - attach_start + 1
                end_frame = fr_s[attach_end_i]
                note = "no_detach_found"
            else:
                dwell_frames = detach_frame - attach_start
                end_frame = detach_frame
                note = "ok"

            dwell_frames = max(int(dwell_frames), 0)
            dwell_seconds = float(dwell_frames) / float(fps if fps else 1.0)

            # --- media score sui frame dell'intervallo ---
            mean_score_pct = np.nan
            if df_score is not None and start_i is not None:
                run_frames = [f for f in fr_s[start_i:] if f <= end_frame]
                if run_frames:
                    s = (df_score[(df_score["track_id"].astype(int) == tid) &
                                  (df_score["frame0"].isin(run_frames))]["score"].astype(float))
                    if len(s) > 0:
                        mean_score_pct = float(s.mean() * 100.0)

            results.append({"bubble_id": tid,
                            "attach_start_frame": attach_start,
                            "detach_frame": detach_frame,
                            "dwell_frames": dwell_frames,
                            "dwell_seconds": dwell_seconds,
                            "n_frames_tracked": len(fr_list),
                            "n_unknown": n_unknown,
                            "mean_score_pct": mean_score_pct,
                            "note": note})

        if out_csv is None:
            out_csv = os.path.join(self.savefolder, f'dwell_{self.extension}.csv')

        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                "bubble_id", "attach_start_frame", "detach_frame",
                "dwell_frames", "dwell_seconds", "n_frames_tracked",
                "n_unknown", "mean_score_pct", "note"
            ])
            w.writeheader()
            for r in results: w.writerow(r)

        print(f"[ComputeAttachDwell] salvato: {out_csv}  (fps={fps})")
        return results