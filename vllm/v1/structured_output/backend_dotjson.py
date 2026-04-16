# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Proprietary
# SPDX-FileCopyrightText: Copyright dottxt

from dataclasses import dataclass

import torch
from dotjson_vllm import DotJsonBackend as _DotJsonBackend
from dotjson_vllm import DotJsonGrammar as _DotJson_Grammar
from transformers import PreTrainedTokenizerFast

from vllm.sampling_params import SamplingParams
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import get_structured_output_key


def get_bitmask_shape(batch_size: int, vocab_size: int) -> tuple[int, int]:
    return (batch_size, (vocab_size + 31) // 32)


def validate_dotjson_grammar(sampling_params: SamplingParams) -> None:
    if sampling_params.structured_outputs is None:
        return
    request_type, grm = get_structured_output_key(sampling_params.structured_outputs)
    if request_type == StructuredOutputOptions.JSON:
        response = _DotJsonBackend.validate_grammar(
            grm,
            sampling_params.structured_outputs.disable_any_whitespace,
            sampling_params.structured_outputs.disable_additional_properties,
        )
        if response:
            raise ValueError(response)
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        return
    else:
        raise ValueError(f"dotJSON does not support {request_type}")


@dataclass
class DotJsonBackend(StructuredOutputBackend):
    def __post_init__(self):
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "dotJSON only supports fast tokenizers. "
                f"Received: {type(self.tokenizer)}"
            )

        num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            and self.vllm_config.speculative_config.num_speculative_tokens is not None
            else 0
        )
        disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )
        disable_additional_properties = (
            self.vllm_config.structured_outputs_config.disable_additional_properties
        )
        vocab_size = max(
            self.vllm_config.model_config.model_arch_config.vocab_size,
            self.vocab_size,
            len(self.tokenizer),
        )
        self._inner = _DotJsonBackend(
            num_speculative_tokens,
            disable_any_whitespace,
            disable_additional_properties,
            vocab_size,
            model=self.vllm_config.model_config.model if self.vllm_config.model_config else "",
            revision=self.vllm_config.model_config.revision
            if self.vllm_config.model_config and self.vllm_config.model_config.revision is not None
            else "",
            tokenizer=self.tokenizer,
            vocabulary_cache_directory_path=self.vllm_config.cache_config.vocabulary_cache_directory_path,
        )

    def compile_grammar(self, request_type: StructuredOutputOptions, grammar_spec: str):
        if request_type == StructuredOutputOptions.JSON:
            return DotJsonGrammar(_inner=self._inner.compile_grammar(grammar_spec))

        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            return DotJsonGrammar(_inner=self._inner.compile_grammar(None))
        else:
            raise ValueError(f"dotJSON does not support {request_type}")

    def allocate_token_bitmask(self, max_num_seqs: int):
        return self._inner.allocate_token_bitmask(max_num_seqs)

    def destroy(self):
        self._inner.destroy()


@dataclass
class DotJsonGrammar(StructuredOutputGrammar):
    _inner: _DotJson_Grammar

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """
        Determines whether the provided tokens are accepted for the
        given request.

        Args:
            request_id (str): The unique identifier for the request.
            tokens (list[int]): A list of token IDs to evaluate.

        Returns:
            bool: True if the tokens are accepted, False otherwise.
        """
        return self._inner.accept_tokens(request_id, tokens)

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """
        Validates the provided tokens against the grammar.
        Will not advance the FSM.

        Args:
            tokens (list[int]): A list of token IDs to validate.

        Returns:
            list[int]: A list of accepted token IDs. Will be a prefix
                of the input tokens, and empty if none are accepted.
        """
        return self._inner.validate_tokens(tokens)

    def rollback(self, num_tokens: int) -> None:
        """
        Rolls back the state of the grammar by a specified number of tokens.
        Will also revert counters for the number of processed tokens.

        Args:
            num_tokens (int): The number of tokens to roll back.
        """
        self._inner.rollback(num_tokens)

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        """
        Fills the bitmask for a specific batch index.

        Args:
            bitmask (torch.Tensor): The bitmask to fill
            batch_index (int): The index in the bitmask to fill
        """

        self._inner.fill_bitmask(bitmask, batch_index)

    def is_terminated(self) -> bool:
        """
        Checks whether the structured output process has terminated.

        Returns:
            bool: True if the process is terminated, False otherwise.
        """
        return self._inner.terminated

    def reset(self):
        """
        Resets the state of the structured output grammar.
        """
        self._inner.reset()
