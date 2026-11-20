"""Assemble data/h4_delta_samples.csv (v4) from the two external sources.
Inputs : data/wosis_layers_0_100cm.csv  (wosis_extraction.py; layers wholly within 0-100 cm)
         data/bis_nederland_samples.csv (BIS Nederland samples, unchanged from v3)
Rules  : drop BIS samples whose sand+silt+clay lies outside 98-102 %;
         drop exact duplicates (same delta, coordinates, depths, SOC, pH and texture),
         which arise where one profile is stored under two WoSIS profile IDs."""
import pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parent.parent / "data"
KEY = ["delta", "lat", "lon", "depth_top_cm", "depth_bot_cm", "soc", "ph", "clay", "silt", "sand"]
w = pd.read_csv(D / "wosis_layers_0_100cm.csv"); w["source"] = "WoSIS"
b = pd.read_csv(D / "bis_nederland_samples.csv")
t = b.clay + b.silt + b.sand; ok = (t >= 98) & (t <= 102)
print(f"BIS samples {len(b)}; texture sum outside 98-102 %: {(~ok).sum()} dropped")
d = pd.concat([w, b[ok]], ignore_index=True); n = len(d)
d = d.drop_duplicates(KEY); print(f"exact duplicates dropped: {n - len(d)}")
d["profile_id"] = d["profile_id"].astype("Int64")
d[KEY + ["source", "profile_id"]].to_csv(D / "h4_delta_samples.csv", index=False)
print(f"wrote h4_delta_samples.csv: {len(d)} samples; " + str(d.source.value_counts().to_dict()))
