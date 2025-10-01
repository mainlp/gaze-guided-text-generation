import random
import warnings
from pathlib import Path
from typing import Generator

import pandas as pd


class GazeTextDataset:
    def __init__(
        self,
        gaze_data: pd.DataFrame,
        texts: dict[str, str],
        gaze_label_mean: float | None = None,
        gaze_label_std: float | None = None,
    ):
        self.gaze_data = gaze_data
        self.texts = texts
        self.gaze_label_mean = gaze_label_mean
        self.gaze_label_std = gaze_label_std

    @classmethod
    def load(
        cls,
        path: str | Path,
        reading_measure: callable,
        outlier_zscore: float | None = None,
        dataset_prefix: str | None = None,
        **kwargs,
    ):
        path = Path(path)
        fixations = pd.read_csv(path / "fixations.csv")
        aois = pd.read_csv(path / "aois.csv")

        # Filter fixations and AOIs based on the provided keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, set):
                fixations = fixations[fixations[key].isin(value)]
                aois = aois[aois[key].isin(value)]
            else:
                fixations = fixations[fixations[key] == value]
                aois = aois[aois[key] == value]

        # Merge fixations and AOIs
        merged = fixations.merge(aois, on=["lang", "text_id", "aoi_index"])

        # Add dataset and language prefix to text and subject IDs
        merged["text_id"] = merged.apply(
            lambda row: f"{row['lang']}__{row['text_id']}", axis=1
        )
        merged["subject_id"] = merged.apply(
            lambda row: f"{row['lang']}__{row['subject_id']}", axis=1
        )
        if dataset_prefix is not None:
            merged["text_id"] = merged.apply(
                lambda row: f"{dataset_prefix}__{row['text_id']}", axis=1
            )
            merged["subject_id"] = merged.apply(
                lambda row: f"{dataset_prefix}__{row['subject_id']}", axis=1
            )

        # Compute reading measures in each trial
        gaze_labels = (
            merged.sort_values("fixation_index")
            .groupby(["text_id", "subject_id"])
            .apply(reading_measure, include_groups=False)
        )

        if outlier_zscore is not None:
            # Clamp outliers within +/- outlier_zscore standard deviations
            zscores = (gaze_labels - gaze_labels.mean()) / gaze_labels.std()
            gaze_labels[zscores > outlier_zscore] = gaze_labels[
                zscores <= outlier_zscore
            ].max()
            gaze_labels[zscores < -outlier_zscore] = gaze_labels[
                zscores >= -outlier_zscore
            ].min()

        # Average reading measures across subjects
        gaze_labels = gaze_labels.groupby(["text_id", "aoi_index"]).mean()
        merged = merged.groupby(["text_id", "aoi_index"]).agg(
            aoi_text=("aoi_text", "first"),
            aoi_text_ws_after=("aoi_text_ws_after", "first"),
        )
        merged["gaze_label"] = gaze_labels
        merged = merged.reset_index()

        # Combine AOI texts into single strings, add aoi_start and aoi_end columns
        texts = {}
        merged = merged.sort_values("aoi_index")
        for text_id, text_aois in merged.groupby("text_id", observed=True):
            text = ""
            aoi_start = []
            aoi_end = []
            for _, row in text_aois.iterrows():
                aoi_start.append(len(text))
                text += row["aoi_text"]
                aoi_end.append(len(text))
                if row["aoi_text_ws_after"]:
                    text += " "
            texts[text_id] = text
            merged.loc[merged["text_id"] == text_id, "aoi_start"] = aoi_start
            merged.loc[merged["text_id"] == text_id, "aoi_end"] = aoi_end

        # Clean up the DataFrame
        merged = merged[
            ["text_id", "aoi_index", "aoi_start", "aoi_end", "aoi_text", "gaze_label"]
        ]
        merged["aoi_start"] = merged["aoi_start"].astype(int)
        merged["aoi_end"] = merged["aoi_end"].astype(int)
        merged = merged.sort_values(["text_id", "aoi_index"])
        merged = merged.reset_index(drop=True)

        return cls(merged, texts)

    def normalize_gaze_labels(
        self, mean: float | None = None, std: float | None = None
    ) -> "GazeTextDataset":
        assert (
            self.gaze_label_mean is None and self.gaze_label_std is None
        ), "Gaze labels are already normalized"
        gaze_labels = self.gaze_data["gaze_label"]
        if mean is None:
            mean = gaze_labels.mean()
        if std is None:
            std = gaze_labels.std()
        self.gaze_data["gaze_label"] = (gaze_labels - mean) / std
        self.gaze_label_mean = mean
        self.gaze_label_std = std
        return self

    def denormalize_gaze_labels(self, gaze_labels):
        assert (
            self.gaze_label_mean is not None and self.gaze_label_std is not None
        ), "Gaze labels are not normalized"
        return gaze_labels * self.gaze_label_std + self.gaze_label_mean

    def random_split(
        self, ratio: float, seed: int = 42
    ) -> tuple["GazeTextDataset", "GazeTextDataset"]:
        num_split = int(len(self.texts) * ratio)
        text_ids = list(self.texts.keys())
        random.seed(seed)
        random.shuffle(text_ids)
        text_ids_set1 = text_ids[:num_split]
        text_ids_set2 = text_ids[num_split:]
        return (
            GazeTextDataset(
                self.gaze_data[self.gaze_data["text_id"].isin(text_ids_set1)],
                {text_id: self.texts[text_id] for text_id in text_ids_set1},
                self.gaze_label_mean,
                self.gaze_label_std,
            ),
            GazeTextDataset(
                self.gaze_data[self.gaze_data["text_id"].isin(text_ids_set2)],
                {text_id: self.texts[text_id] for text_id in text_ids_set2},
                self.gaze_label_mean,
                self.gaze_label_std,
            ),
        )

    def iter_texts(self) -> Generator[tuple[str, str, pd.DataFrame], None, None]:
        for text_id, text_aois in self.gaze_data.sort_values("aoi_index").groupby(
            "text_id", observed=True
        ):
            text = self.texts[text_id]
            yield text_id, text, text_aois

    def __add__(self, other: "GazeTextDataset") -> "GazeTextDataset":
        if getattr(self, "gaze_label_mean", None) != getattr(
            other, "gaze_label_mean", None
        ):
            warnings.warn(
                "Datasets have been normalized differently."
                " You will not be able to denormalize gaze labels."
            )
            gaze_label_mean = None
            gaze_label_std = None
        else:
            gaze_label_mean = self.gaze_label_mean
            gaze_label_std = self.gaze_label_std

        texts = self.texts | other.texts
        gaze_data = pd.concat([self.gaze_data, other.gaze_data], ignore_index=True)
        return GazeTextDataset(gaze_data, texts, gaze_label_mean, gaze_label_std)
