import re
import sys
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wordfreq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from .dataset import GazeTextDataset


class GazeModel(ABC):
    @abstractmethod
    def fit(
        self, train_dataset: GazeTextDataset, dev_dataset: GazeTextDataset, **kwargs
    ):
        pass

    @abstractmethod
    def predict(self, texts: list[str]) -> list[float]:
        pass

    def predict_aois(self, text: str, aoi_ends: list[int]) -> list[float]:
        preds = []
        previous_total_pred = 0
        for aoi_end in aoi_ends:
            text_up_to_aoi = text[:aoi_end]
            total_pred = self.predict([text_up_to_aoi])[0]

            # Gaze models predict the sum of gaze labels for the entire text,
            # so we need to subtract the previous prediction from the current one
            pred = total_pred - previous_total_pred
            preds.append(pred)
            previous_total_pred = total_pred
        return preds

    def evaluate(self, dataset: GazeTextDataset) -> dict[str, float]:
        preds = []
        labels = []
        for text_id, text_aois in dataset.gaze_data.sort_values("aoi_index").groupby(
            "text_id", observed=True
        ):
            text = dataset.texts[text_id]
            aoi_ends = text_aois["aoi_end"]
            preds.extend(self.predict_aois(text, aoi_ends))
            labels.extend(text_aois["gaze_label"])

        return {
            "preds": preds,
            "labels": labels,
            "mse": mean_squared_error(labels, preds),
            "mae": mean_absolute_error(labels, preds),
            "rmse": root_mean_squared_error(labels, preds),
            "r2": r2_score(labels, preds),
        }


class TransformerGazeTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: GazeTextDataset, tokenizer: PreTrainedTokenizer):
        # Preprocess data
        self.texts = [
            self.preprocess_text(text_aois, dataset.texts[text_id], tokenizer)
            for text_id, text_aois in dataset.gaze_data.sort_values(
                "aoi_index"
            ).groupby("text_id", observed=True)
        ]

    @staticmethod
    def preprocess_text(
        text_aois: pd.DataFrame, text: str, tokenizer: PreTrainedTokenizer
    ) -> dict[str, torch.Tensor]:
        # Tokenize text
        text_encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)

        # Remove batch dimension
        for key, value in text_encoded.items():
            text_encoded[key] = value[0]

        # Align AOIs with tokens
        aoi_index = 0
        aoi_num_tokens = 0
        gaze_labels = []
        for token, (token_start, token_end) in zip(
            text_encoded["input_ids"], text_encoded["offset_mapping"]
        ):
            aoi = text_aois.iloc[aoi_index]
            aoi_start = aoi["aoi_start"]
            aoi_end = aoi["aoi_end"]
            if token_start < aoi_end and token_end > aoi_start:
                # Token is within current AOI
                aoi_num_tokens += 1
            elif token_start >= aoi_end:
                # Token is outside current AOI -> current AOI is complete
                for _ in range(aoi_num_tokens):
                    gaze_labels.append(aoi["gaze_label"] / aoi_num_tokens)
                aoi_index += 1
                aoi_num_tokens = 1
            else:
                # No overlapping AOI
                raise ValueError(f"{token=!r} {token_start=!r} {token_end=!r} {aoi=!r}")
        # Last AOI
        for _ in range(aoi_num_tokens):
            gaze_labels.append(aoi["gaze_label"] / aoi_num_tokens)

        text_encoded["gaze_label"] = torch.tensor(gaze_labels, dtype=torch.float32)
        assert text_encoded["input_ids"].size() == text_encoded["gaze_label"].size()
        text_encoded.pop("offset_mapping")
        return text_encoded

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def pad_batch(batch):
    return {
        key: nn.utils.rnn.pad_sequence(
            [sample[key] for sample in batch],
            batch_first=True,
            padding_value=0,
        )
        for key in batch[0].keys()
    }


class CausalTransformerGazeModel(nn.Module, GazeModel):
    def __init__(self, transformer, tokenizer, dropout=0.5, freeze=False):
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "left"
        if freeze:
            self.transformer.requires_grad_(False)
        else:
            self.transformer.requires_grad_(True)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.transformer.config.hidden_size
        self.gaze_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        transformer = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(transformer, tokenizer, **kwargs)

    @property
    def device(self):
        return self.transformer.device

    def forward(self, input_ids, **kwargs):
        outputs = self.transformer(input_ids, output_hidden_states=True, **kwargs)
        dropout_outputs = self.dropout(outputs.hidden_states[-1])
        gaze_outputs = self.gaze_head(dropout_outputs).squeeze(-1)
        return gaze_outputs

    def fit(
        self,
        train_dataset: GazeTextDataset,
        dev_dataset: GazeTextDataset,
        *,
        batch_size: int = 1,
        max_epochs: int = float("inf"),
        patience: int = 0,
        learning_rate: float = 0.001,
    ):
        train_dataset = TransformerGazeTextDataset(train_dataset, self.tokenizer)
        dev_dataset = TransformerGazeTextDataset(dev_dataset, self.tokenizer)

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch
        )
        dev_dataloader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_batch
        )

        best_dev_loss = float("inf")
        num_epochs_without_improvement = 0
        best_checkpoint = deepcopy(self.state_dict())
        epoch = 0
        while True:
            self.train()
            train_loss = 0
            for batch in train_dataloader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                gaze_labels = batch.pop("gaze_label")
                optimizer.zero_grad()
                gaze_outputs = self(**batch)
                gaze_outputs = gaze_outputs * batch["attention_mask"]
                batch_loss = (
                    nn.functional.mse_loss(gaze_outputs, gaze_labels, reduction="sum")
                    / batch["attention_mask"].sum()
                )
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item()
            train_loss /= len(train_dataloader)

            self.eval()
            dev_loss = 0
            for batch in dev_dataloader:
                batch = {key: value.cuda() for key, value in batch.items()}
                gaze_labels = batch.pop("gaze_label")
                with torch.no_grad():
                    gaze_outputs = self(**batch)
                gaze_outputs = gaze_outputs * batch["attention_mask"]
                dev_loss += (
                    nn.functional.mse_loss(gaze_outputs, gaze_labels, reduction="sum")
                    / batch["attention_mask"].sum()
                ).item()
            dev_loss /= len(dev_dataloader)

            epoch += 1
            print(
                f"{epoch} epochs | {train_loss = :.3f} | {dev_loss = :.3f}",
                file=sys.stderr,
            )

            if epoch >= max_epochs:
                break
            if dev_loss > best_dev_loss:
                num_epochs_without_improvement += 1
                if num_epochs_without_improvement > patience:
                    break
            else:
                best_checkpoint = deepcopy(self.state_dict())
                best_dev_loss = dev_loss
                num_epochs_without_improvement = 0
        self.load_state_dict(best_checkpoint)

    def predict(self, texts: list[str]) -> list[float]:
        texts = [re.sub(r"\s+", " ", text).strip() for text in texts]
        text_encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )
        self.eval()
        with torch.no_grad():
            gaze_outputs = self(**text_encoded)
            gaze_outputs = gaze_outputs * text_encoded["attention_mask"]
            gaze_scores = gaze_outputs.sum(dim=1)
        return gaze_scores.cpu().tolist()


class LinearRegressionGazeModel(GazeModel):
    def __init__(self, lang: str = "en", max_spillover: int = 2):
        self.lang = lang
        self.max_spillover = max_spillover

        self.linear_regression = LinearRegression()
        self.feature_means = None

    def preprocess_dataset(self, dataset: GazeTextDataset) -> tuple[np.array, np.array]:
        X = []
        y = []
        for text_id, text_aois in dataset.gaze_data.sort_values("aoi_index").groupby(
            "text_id", observed=True
        ):
            text = dataset.texts[text_id]
            words, word_starts, word_ends = self.tokenize(text)

            # Align AOIs with words
            aoi_index = 0
            aoi_num_words = 0
            gaze_labels = []
            for word, word_start, word_end in zip(words, word_starts, word_ends):
                aoi = text_aois.iloc[aoi_index]
                aoi_start = aoi["aoi_start"]
                aoi_end = aoi["aoi_end"]
                if word_start < aoi_end and word_end > aoi_start:
                    # Word is within current AOI
                    aoi_num_words += 1
                elif word_start >= aoi_end:
                    # Word is outside current AOI -> current AOI is complete
                    for _ in range(aoi_num_words):
                        gaze_labels.append(aoi["gaze_label"] / aoi_num_words)
                    aoi_index += 1
                    aoi_num_words = 1
                else:
                    # No overlapping AOI
                    raise ValueError(
                        f"{word=!r} {word_start=!r} {word_end=!r} {aoi=!r}"
                    )
            # Last AOI
            for _ in range(aoi_num_words):
                gaze_labels.append(aoi["gaze_label"] / aoi_num_words)

            # Add features and labels
            X.extend(self.featurize(words, replace_nans=False))
            y.extend(gaze_labels)
        X = np.array(X)
        y = np.array(y)

        return X, y

    def tokenize(self, text: str) -> tuple[list[str], list[int], list[int]]:
        matches = list(re.finditer(r"\w+", text))
        words = [match.group(0) for match in matches]
        start_indices = [match.start() for match in matches]
        end_indices = [match.end() for match in matches]
        return words, start_indices, end_indices

    def featurize(self, words: list[str], replace_nans: bool = True) -> np.array:
        features = []
        epsilon = 1e-8
        if len(words) == 0:
            features = np.empty((1, 2 * (self.max_spillover + 1)))
            features.fill(np.nan)
        for i in range(len(words)):
            lengths = []
            frequencies = []
            for offset in range(self.max_spillover + 1):
                if i - offset >= 0:
                    length = len(words[i - offset])
                    frequency = wordfreq.word_frequency(words[i - offset], "en")
                    if frequency == 0:
                        frequency = epsilon
                    frequency = np.log(frequency)
                else:
                    length = np.nan
                    frequency = np.nan
                lengths.append(length)
                frequencies.append(frequency)
            features.append(lengths + frequencies)
        features = np.array(features)

        # Replace NAs with means
        if replace_nans:
            assert self.feature_means is not None, "Feature means are not set"
            features = np.where(np.isnan(features), self.feature_means, features)

        return features

    def fit(self, train_dataset: GazeTextDataset, dev_dataset: GazeTextDataset):
        # Preprocess data
        X, y = self.preprocess_dataset(train_dataset)

        # Replace NAs with means
        self.feature_means = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), self.feature_means, X)

        # Fit model
        self.linear_regression.fit(X, y)

    def predict(self, texts: list[str]) -> list[float]:
        # Preprocess data
        Xs = []
        for text in texts:
            words, _, _ = self.tokenize(text)
            Xs.append(self.featurize(words))

        # Predict gaze labels
        ys = []
        for X in Xs:
            y = self.linear_regression.predict(X).sum()
            ys.append(y)
        return ys
