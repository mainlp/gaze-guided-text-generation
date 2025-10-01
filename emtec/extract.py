# %%
import os
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# %%
fixation_data = pd.read_csv(f"{SCRIPT_DIR}/raw/fixations-corrected.csv", sep="\t")

# Remove fixations with out-of-range word indices (probably not on stimulus?)
fixation_data = fixation_data[fixation_data["word"] != "."]

fixation_data["text_id"] = fixation_data.apply(
    lambda row: f"{row['model']}_{row['decoding_strategy']}_{row['item_id']}", axis=1
)
fixation_data["fixation_index"] = fixation_data["fixation_index"] - 1

# %%
stimulus_data = pd.read_csv(f"{SCRIPT_DIR}/raw/stimuli.csv", sep="\t")
stimulus_data["text_id"] = stimulus_data.apply(
    lambda row: f"{row['model']}_{row['decoding_strategy']}_{row['item_id']}", axis=1
)
stimulus_data["words"] = stimulus_data["gen_seq_trunc"].str.strip().str.split()
stimulus_data["word_indices"] = stimulus_data["words"].apply(
    lambda words: list(range(len(words)))
)

# %%
# Make sure that the word indices match up with our tokenization
joined_data = fixation_data.merge(stimulus_data, on="text_id")
for _, row in joined_data.iterrows():
    assert row["word"] == row["words"][row["word_id"]], (
        row["text_id"],
        row["word"],
        row["words"][row["word_id"]],
    )

# %%
fixation_data["lang"] = "en"
fixation_data = fixation_data.rename(columns={"word_id": "aoi_index"})

# %%
aoi_data = stimulus_data[["text_id", "words", "word_indices"]]
aoi_data["words_ws_after"] = aoi_data["words"].apply(
    lambda words: [True] * (len(words) - 1) + [False]
)
aoi_data = aoi_data.explode(["words", "word_indices", "words_ws_after"]).reset_index(
    drop=True
)
aoi_data = aoi_data.rename(
    columns={
        "words": "aoi_text",
        "word_indices": "aoi_index",
        "words_ws_after": "aoi_text_ws_after",
    }
)
aoi_data["lang"] = "en"

# %%
os.makedirs("fixations", exist_ok=True)

fixation_data = fixation_data[
    [
        "lang",
        "text_id",
        "aoi_index",
        "subject_id",
        "fixation_index",
        "fixation_duration",
    ]
]
fixation_data.to_csv(f"{SCRIPT_DIR}/fixations/fixations.csv", index=False)

aoi_data = aoi_data[["lang", "text_id", "aoi_index", "aoi_text", "aoi_text_ws_after"]]
aoi_data.to_csv(f"{SCRIPT_DIR}/fixations/aois.csv", index=False)
