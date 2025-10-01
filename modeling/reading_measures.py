import pandas as pd


def mean_fixation_duration(trial):
    return trial.groupby("aoi_index")["fixation_duration"].mean()


def total_fixation_duration(trial):
    return trial.groupby("aoi_index")["fixation_duration"].sum()


def first_fixation_duration(trial):
    return trial.groupby("aoi_index")["fixation_duration"].first()


def first_pass_gaze_duration(trial):
    durations = pd.Series(
        index=pd.Index(trial["aoi_index"].unique(), name="aoi_index"),
        data=0,
        dtype=float,
    )
    latest_aoi_index = -1
    regressed = False
    for _, row in trial.iterrows():
        if row["aoi_index"] > latest_aoi_index:
            latest_aoi_index = row["aoi_index"]
            regressed = False
        elif row["aoi_index"] < latest_aoi_index:
            regressed = True
        if not regressed:
            durations[latest_aoi_index] += row["fixation_duration"]
    return durations


def go_past_duration(trial):
    durations = pd.Series(
        index=pd.Index(trial["aoi_index"].unique(), name="aoi_index"),
        data=0,
        dtype=float,
    )
    latest_aoi_index = -1
    for _, row in trial.iterrows():
        if row["aoi_index"] > latest_aoi_index:
            latest_aoi_index = row["aoi_index"]
        durations[latest_aoi_index] += row["fixation_duration"]
    return durations
