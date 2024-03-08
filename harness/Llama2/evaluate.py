import sys  # NOQA
sys.path.append('../')  # NOQA


import os
from dotenv import load_dotenv
from TruthfulQA.truthfulqa import utilities, evaluate, metrics
from llama2_tpu import TpuLLama2
import numpy as np
import logging
import pandas as pd

from TruthfulQA.truthfulqa.models import MC_calcs, set_columns
load_dotenv()

questions = utilities.load_questions(os.getenv('EVA_QUESTION_PATH'))
outpath = os.getenv('EVA_OUTPUT_PATH')


BEST_COL = 'Best Answer'
ANSWER_COL = 'Correct Answers'
INCORRECT_COL = 'Incorrect Answers'


def run_llama2_tpu(frame, engine=None, tag='tpu', preset='qa', model=None, tokenizer=None, verbose=True, device=None, cache_dir=None):

    if tag not in frame.columns:
        frame[tag] = ''
    frame[tag].fillna('', inplace=True)
    frame[tag] = frame[tag].astype(str)

    model = TpuLLama2()
    ctx = model.Llama2_with_devid_and_model(0)

    for idx in frame.index:

        print(idx)
        if pd.isnull(frame.loc[idx, tag]) or not len(frame.loc[idx, tag]):

            prompt = utilities.format_prompt(
                frame.loc[idx], preset, format='UQA')

            resp = model.Llama2_complete(ctx, prompt)

            # print(resp)

            frame.loc[idx, tag] = resp
        else:
            frame.loc[idx, tag] = ''

    return frame


def run_metrics(model_key, questions, metric=None):

    questions = metrics.run_BLEURT(
        model_key, questions)
    questions = metrics.run_bleu_and_rouge(
        model_key, questions)
    questions = metrics.run_end2end_GPT3(
        model_key, questions)

    utilities.save_questions(questions, outpath)

    return


def run_probs(frame, engine, tag, preset='qa', model=None, tokenizer=None, device=None, cache_dir=None):
    """Runs multiple-choice metrics for autoregressive HuggingFace models (GPT-2, GPT-Neo)"""

    set_columns(tag, frame)

    model = TpuLLama2()
    ctx = model.Llama2_with_devid_and_model(0)

    # if model is None:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         engine, return_dict_in_generate=True, cache_dir=cache_dir).to(device)
    #     model.eval()
    # if tokenizer is None:
    #     tokenizer = AutoTokenizer.from_pretrained(engine, cache_dir=cache_dir)

    for idx in frame.index:
        if pd.isnull(frame.loc[idx, '{0} lprob max'.format(tag)]):

            # check that answer exists
            if pd.isnull(frame.loc[idx, INCORRECT_COL]):
                logging.warn(
                    "References missing for {0}!".format(idx), stacklevel=2)
                continue
            if not len(frame.loc[idx, INCORRECT_COL]):
                logging.warn(
                    "References missing for {0}!".format(idx), stacklevel=2)
                continue

            # reference answers
            ref_best = utilities.format_best(frame.loc[idx, BEST_COL])
            ref_true = utilities.split_multi_answer(
                frame.loc[idx, ANSWER_COL])
            ref_false = utilities.split_multi_answer(
                frame.loc[idx, INCORRECT_COL])

            scores_true = []
            scores_false = []

            input_prompt = utilities.format_prompt(
                frame.loc[idx], preset, format='general')

            for temp_ans in ref_true:
                # append the current answer choice to the prompt
                prompt = utilities.format_prompt_with_answer_strings(frame.loc[idx, 'Question'],
                                                                     temp_ans,
                                                                     preset,
                                                                     format='general')
                input_ids = tokenizer(
                    input_prompt, return_tensors="pt").input_ids.to(device)
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt").input_ids.to(device)

                outputs = model(prompt_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[input_ids.shape[-1] - 1: -1, :]
                prompt_ids = prompt_ids[0, input_ids.shape[-1]:]

                # get logprobs for each token in the answer
                log_probs = outputs[range(
                    outputs.shape[0]), prompt_ids.squeeze(0)]
                log_probs = log_probs[3:]  # drop the '\nA:' prefix

                scores_true.append(log_probs.sum().item())

            for temp_ans in ref_false:
                # append the current answer choice to the prompt
                prompt = utilities.format_prompt_with_answer_strings(frame.loc[idx, 'Question'],
                                                                     temp_ans,
                                                                     preset,
                                                                     format='general')
                input_ids = tokenizer(
                    input_prompt, return_tensors="pt").input_ids.to(device)
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt").input_ids.to(device)

                outputs = model(prompt_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[input_ids.shape[-1] - 1: -1, :]
                prompt_ids = prompt_ids[0, input_ids.shape[-1]:]

                # get logprobs for each token in the answer
                log_probs = outputs[range(
                    outputs.shape[0]), prompt_ids.squeeze(0)]
                log_probs = log_probs[3:]  # drop the '\nA:' prefix

                scores_false.append(log_probs.sum().item())

            MC_calcs(tag, frame, idx, scores_true,
                     scores_false, ref_true, ref_best)

    return frame


def main():

    load_dotenv()

    # run_llama2_tpu(questions)

    # utilities.save_questions(questions, outpath)
    questions = utilities.load_questions('./build/result.csv')
    run_metrics('tpu', questions)

    results = evaluate.format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    results = results[results['Metric'].isin(['MC1', 'MC2',
                                              'bleu acc',
                                              'rouge1 acc',
                                              'BLEURT acc',
                                              'GPT-judge acc',
                                              'GPT-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')
    results.to_csv('summary.csv')
    return


if __name__ == "__main__":
    main()
