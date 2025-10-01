import argparse
import json
import os
import pickle

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modeling.generation import GazeControlledBeamSearch
from modeling.gaze_models import CausalTransformerGazeModel


def generate(prompts_filename, language_model_name, gaze_model_name, gaze_weight, beam_size, gpu, verbose):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    language_model_tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    language_model = AutoModelForCausalLM.from_pretrained(language_model_name).cuda()

    if gaze_model_name == "trf":
        gaze_model = CausalTransformerGazeModel.from_pretrained(
            "openai-community/gpt2"
        ).cuda()
        gaze_model.load_state_dict(torch.load("models/trf_gaze_model.pt"))
    elif gaze_model_name == "lr":
        with open("models/lr_gaze_model.pkl", "rb") as f:
            gaze_model = pickle.load(f)
    else:
        raise ValueError(f"Unknown gaze model name: {gaze_model_name}")

    with open(prompts_filename) as f:
        prompts = [json.loads(line) for line in f]

    beam_search = GazeControlledBeamSearch(
        language_model,
        language_model_tokenizer,
        gaze_model,
    )

    for prompt in prompts:
        input_text = language_model_tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a short story based on the following title and prompt.\n"
                        f"Title: {prompt['title']}\n"
                        f"Prompt: {prompt['prompt']}\n\n"
                        "The story should not be longer than 500 words. "
                        "Keep in mind that the reader will not see the prompt, only the story itself. "
                        "Do not include the title."
                    ),
                },
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        output_text, token_score, gaze_score = beam_search.generate(
            input_text,
            gaze_weight=gaze_weight,
            max_length=800,
            beam_size=beam_size,
            ignore_prompt=True,
            verbose=verbose,
        )

        print(
            json.dumps(
                {
                    **prompt,
                    "gaze_weight": gaze_weight,
                    "input_text": input_text,
                    "output_text": output_text,
                    "token_score": token_score,
                    "gaze_score": gaze_score,
                },
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        type=str,
        dest="prompts_filename",
        required=True,
        help="Path to the prompts file.",
    )
    parser.add_argument(
        "--language-model",
        type=str,
        dest="language_model_name",
        required=True,
        help="Name of the language model.",
    )
    parser.add_argument(
        "--gaze-model",
        type=str,
        dest="gaze_model_name",
        required=True,
        help="Name of the gaze model.",
    )
    parser.add_argument(
        "--gaze-weight",
        type=float,
        required=True,
        help="Weight for the gaze scores.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        help="Beam size for the generation.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use for generation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print best beam during generation.",
    )

    args = parser.parse_args()

    generate(
        args.prompts_filename,
        args.language_model_name,
        args.gaze_model_name,
        args.gaze_weight,
        args.beam_size,
        args.gpu,
        args.verbose,
    )
