"""PickleballVision dashboard. Run after app.py:  streamlit run dashboard.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pbv.draw import name

st.set_page_config(page_title="PickleballVision", page_icon="🏓", layout="wide")
runs = sorted(p.parent for p in Path("outputs").glob("*/stats.json"))
if not runs:
    st.info("No results yet. Run `python app.py <video>` first.")
    st.stop()
run = st.sidebar.selectbox("Match", runs, format_func=lambda p: p.name)
s = json.loads((run / "stats.json").read_text())
shots = s["shots"]

st.title(f"🏓 {run.name}")
if s.get("mode") == "pov":
    st.caption("POV mode: body-worn camera. Court positions, heatmaps, speeds and kitchen faults need fixed mode.")
c = st.columns(4)
c[0].metric("Duration", f"{s['frames'] / s['fps']:.0f} s")
c[1].metric("Rallies", len(s["rallies"]))
c[2].metric("Shots", len(shots))
c[3].metric("Faults", sum(len(x["faults"]) for x in shots))

if (run / "annotated.mp4").exists():
    st.video(str(run / "annotated.mp4"))

st.subheader("Players")
pids = list(s["players"]) + [p for p in s["shot_summary"] if p not in s["players"]]
st.dataframe(pd.DataFrame([{"player": name(pid), **s["players"].get(pid, {}), **s["shot_summary"].get(pid, {})}
                           for pid in pids]).fillna(0), hide_index=True)

if shots:
    st.subheader("Shot mix")
    df = pd.DataFrame(shots)
    st.bar_chart(df.assign(player=df["player"].map(name)).pivot_table(
        index="type", columns="player", values="frame", aggfunc="count", fill_value=0))

st.subheader("Court coverage")
maps = sorted(run.glob("heatmap_P*.png"))
if not maps:
    st.caption("Heatmaps need fixed mode.")
for col, m in zip(st.columns(max(len(maps), 1)), maps):
    col.image(str(m), caption=m.stem.removeprefix("heatmap_"))

tab_shots, tab_rallies = st.tabs(["Shot log", "Rallies"])
with tab_shots:
    if shots:
        log = pd.json_normalize(shots).drop(columns=["frame"], errors="ignore")
        log["faults"] = log["faults"].map(", ".join)
        only_faults = st.checkbox("Only shots with faults")
        st.dataframe(log[log["faults"] != ""] if only_faults else log, hide_index=True)
with tab_rallies:
    st.dataframe(pd.DataFrame(s["rallies"]), hide_index=True)
