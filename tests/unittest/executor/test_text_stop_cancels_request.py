# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""A stop string matched only in the detokenized text must cancel the engine request.

The engine matches stop strings by token ids. When the model emits the stop string
inside a differently tokenized piece (stop ``"\n"`` generated as a ``"\n\n"`` token),
only the text-level check in ``DetokenizedGenerationResultBase`` sees it. That check
ends the result for the caller, so the engine request has to be cancelled as well,
or it keeps generating up to ``max_tokens``. CPU-only; no engine needed.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.result import DetokenizedGenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

NEWLINE = 10
DOUBLE_NEWLINE = 11
PIECES = {1: "Hi", 2: " there", NEWLINE: "\n", DOUBLE_NEWLINE: "\n\n"}


class _PieceTokenizer:
    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(PIECES[token_id] for token_id in token_ids)


class _CountingResult(DetokenizedGenerationResultBase):
    abort_calls = 0

    def abort(self):
        self.abort_calls += 1
        super().abort()


def _response(token_ids, finish_reason, is_final):
    result = SimpleNamespace(
        is_final=is_final,
        context_phase_params=None,
        decoding_iter=1,
        cached_tokens=0,
        avg_decoded_tokens_per_iter=None,
        finish_reasons=[finish_reason],
        output_token_ids=[list(token_ids)],
        request_perf_metrics=None,
        sequence_index=0,
        cum_log_probs=None,
        log_probs=None,
        generation_logits=None,
        context_logits=None,
        per_pos_drafted=None,
        per_pos_accepted=None,
        spec_dec_totals=None,
        additional_context_outputs=None,
        additional_generation_outputs=None,
        time_breakdown_metrics=None,
    )
    return SimpleNamespace(has_error=lambda: False, result=result)


def _result(streaming):
    sampling_params = SamplingParams(end_id=0, stop="\n", max_tokens=64)
    # Stands in for SamplingParams._setup tokenizing "\n".
    sampling_params._stop_word_ids = [[NEWLINE]]
    return _CountingResult(0, sampling_params, tokenizer=_PieceTokenizer(), streaming=streaming)


def test_text_stop_before_engine_finish_cancels_request():
    result = _result(streaming=True)

    result._handle_response(
        _response([1, DOUBLE_NEWLINE], tllm.FinishReason.NOT_FINISHED, is_final=False)
    )

    output = result.outputs[0]
    assert output.text == "Hi"
    assert output.finish_reason == "stop"
    assert output.stop_reason == "\n"
    assert result._done
    assert result.abort_calls == 1
    # On a postprocess worker this flag is what the proxy turns into a cancel.
    assert result._aborted


def test_text_stop_cancels_only_once():
    result = _result(streaming=True)
    response = _response([1, DOUBLE_NEWLINE], tllm.FinishReason.NOT_FINISHED, is_final=False)

    result._handle_response(response)
    result._handle_response(_response([2], tllm.FinishReason.NOT_FINISHED, is_final=False))

    assert result.abort_calls == 1


def test_text_stop_in_final_response_does_not_cancel():
    result = _result(streaming=False)

    result._handle_response(_response([1, DOUBLE_NEWLINE], tllm.FinishReason.LENGTH, is_final=True))

    output = result.outputs[0]
    assert output.text == "Hi"
    assert output.finish_reason == "stop"
    assert result.abort_calls == 0
    assert not result._aborted


def test_no_stop_match_does_not_cancel():
    result = _result(streaming=True)

    result._handle_response(_response([1, 2], tllm.FinishReason.NOT_FINISHED, is_final=False))

    assert result.outputs[0].text == "Hi there"
    assert not result._done
    assert result.abort_calls == 0
