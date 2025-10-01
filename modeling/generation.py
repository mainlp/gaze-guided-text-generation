import sys

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .gaze_models import GazeModel


class GazeControlledBeamSearch:
    def __init__(
        self,
        language_model: PreTrainedModel,
        language_model_tokenizer: PreTrainedTokenizer,
        gaze_model: GazeModel,
    ):
        self.language_model = language_model
        self.language_model_tokenizer = language_model_tokenizer
        self.gaze_model = gaze_model
        self.eos_token_id = language_model_tokenizer.eos_token_id

    def generate(
        self,
        input_text,
        beam_size=10,
        max_length=100,
        gaze_weight=0,
        ignore_prompt=False,
        verbose=False,
    ):
        """
        Generate text using gaze-controlled beam search and return the text from the
        best beam, along with that beam's total token and gaze scores.
        
        Generation stops when the best beam produces an EOS token or when `max_length`
        is reached. `ignore_prompt` indicates whether the prompt should be excluded
        from the context for the gaze model.
        """
        self.language_model.eval()

        beam_token_ids = self.language_model_tokenizer(input_text, return_tensors="pt")[
            "input_ids"
        ]
        if ignore_prompt:
            prompt_length = beam_token_ids.size(1)
        else:
            prompt_length = 0
        beam_token_ids = beam_token_ids.to(self.language_model.device)
        beam_token_scores = torch.zeros(1).to(self.language_model.device)
        beam_gaze_scores = torch.zeros(1).to(self.language_model.device)
        beam_finished = torch.zeros(1, dtype=torch.bool).to(self.language_model.device)

        for i in range(max_length):
            with torch.no_grad():
                beam_token_ids, beam_token_scores, beam_gaze_scores, beam_finished = (
                    self.generate_step(
                        beam_token_ids,
                        beam_token_scores,
                        beam_gaze_scores,
                        beam_finished,
                        beam_size,
                        gaze_weight,
                        prompt_length,
                    )
                )

            if verbose:
                best_beam_token_ids = beam_token_ids[0][prompt_length:]
                best_beam_text = self.language_model_tokenizer.decode(
                    best_beam_token_ids
                )
                print(
                    f"Best beam at step {i + 1}:",
                    best_beam_text,
                    file=sys.stderr,
                    flush=True,
                )

            # Stop generating when the best beam is finished
            if beam_finished[0]:
                break

        # Return best beam
        best_beam_token_ids = beam_token_ids[0, prompt_length:]
        best_beam_text = self.language_model_tokenizer.decode(
            best_beam_token_ids, skip_special_tokens=True
        )
        best_beam_token_score = beam_token_scores[0].item()
        best_beam_gaze_score = beam_gaze_scores[0].item()
        return best_beam_text, best_beam_token_score, best_beam_gaze_score

    def generate_step(
        self,
        beam_token_ids,
        beam_token_scores,
        beam_gaze_scores,
        beam_finished,
        beam_size,
        gaze_weight,
        prompt_length,
    ):
        """
        Take the previous generation of beams (token IDs, scores, finished mask) and
        return the next generation of beams (token IDs, scores, finished mask)
        """
        # Predict next token
        output = self.language_model(beam_token_ids)
        logprobs = output.logits[:, -1].log_softmax(-1)

        # For finished beams (where EOS has already been produced), force next token ID
        # to be EOS with probability 1.0 (i.e., continue predicting EOS)
        logprobs[beam_finished] = -torch.inf
        logprobs[beam_finished, self.eos_token_id] = 0

        # Find the top k candidates for each beam based on token score
        beam_topk_token_scores, beam_topk_token_ids = torch.topk(logprobs, beam_size)
        candidate_token_ids = torch.cat(
            [
                beam_token_ids.repeat_interleave(beam_size, dim=0),
                beam_topk_token_ids.view(-1, 1),
            ],
            dim=-1,
        )

        # Add token scores to previous beam scores
        candidate_token_scores = beam_token_scores.repeat_interleave(
            beam_size
        ) + beam_topk_token_scores.view(-1)

        # Check for newly generate EOS tokens, update finished mask
        candidate_finished = beam_finished.repeat_interleave(beam_size)
        candidate_finished |= candidate_token_ids[:, -1] == self.eos_token_id

        # Detokenize and remove prompt
        candidate_texts = self.language_model_tokenizer.batch_decode(
            candidate_token_ids[:, prompt_length:], skip_special_tokens=True
        )

        # Predict gaze for each candidate
        candidate_gaze_scores = self.gaze_model.predict(candidate_texts)
        candidate_gaze_scores = torch.tensor(candidate_gaze_scores).to(
            candidate_token_scores.device
        )

        # Find the top k candidates across all beams based on total (token + gaze) score
        candidate_total_scores = (
            candidate_token_scores + gaze_weight * candidate_gaze_scores
        )
        _, topk_indices = torch.topk(candidate_total_scores, beam_size)
        beam_token_ids = candidate_token_ids[topk_indices]
        beam_token_scores = candidate_token_scores[topk_indices]
        beam_gaze_scores = candidate_gaze_scores[topk_indices]
        beam_finished = candidate_finished[topk_indices]

        return beam_token_ids, beam_token_scores, beam_gaze_scores, beam_finished
