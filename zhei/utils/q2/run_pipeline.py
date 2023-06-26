# Copyright 2020 The Q2 Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import numpy as np
import pandas as pd
import spacy

import zhei.utils.q2.question_generation as qg
import zhei.utils.q2.question_answering as qa
from score import f1_score, clean_text
from tqdm import tqdm
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForQuestionAnswering

INVALID_QUESTION = -1
NO_ANS = '[CLS]'
NO_VALID_QUESTIONS = 'NO_Q'

nlp = spacy.load("en_core_web_sm")


def filter_questions(exp_ans, pred_ans):
    if pred_ans == NO_ANS:
        return 'NO MATCH'
    if clean_text(exp_ans) != clean_text(pred_ans):
        return 'NO MATCH'
    return 'VALID'


def non_personal(question):
    question_tok = nlp(question)
    for tok in question_tok:
        if tok.dep_ == 'nsubj':
            if tok.text.lower() == 'i' or tok.text.lower() == 'you':
                return False
        elif tok.dep_ == 'poss':
            if tok.text.lower() == 'my' or tok.text.lower() == 'your':
                return False
    return True


def single_question_score(question, cand, response, knowledge, qa_model, qa_tokenizer):
    pred_ans = qa.get_answer(question, response, qa_model, qa_tokenizer)

    if filter_questions(cand, pred_ans) == 'VALID':
        knowledge_ans = qa.get_answer(question, knowledge, qa_model, qa_tokenizer)
        if knowledge_ans != NO_ANS:
            return f1_score(cand, knowledge_ans), knowledge_ans
        else:
            return 0, NO_ANS
    else:
        return INVALID_QUESTION, INVALID_QUESTION


def get_response_score(response, knowledge, gen_method, single, remove_personal, qg_model, qg_tokenizer, qa_model, qa_tokenizer):
    f1 = 0
    num_questions = 0

    valid_questions = []
    valid_cands = []
    knowledge_answers = []
    scores = []



    candidates = qg.get_answer_candidates(response)
    for cand in candidates:
        if gen_method == 'greedy':
            questions = [qg.get_question_greedy(cand, response, qg_model, qg_tokenizer)]
        elif gen_method == 'beam':
            questions = qg.get_questions_beam(cand, response, qg_model, qg_tokenizer)
        else:
            questions = qg.get_questions_sample(cand, response, qg_model, qg_tokenizer)

        for question in questions:
            if not remove_personal or non_personal(question):
                question_score, knowledge_ans = single_question_score(question, cand, response, knowledge, qa_model, qa_tokenizer)
                if question_score != INVALID_QUESTION:
                    num_questions += 1
                    f1 += question_score

                    valid_questions.append(question)
                    valid_cands.append(cand)
                    knowledge_answers.append(knowledge_ans)
                    scores.append(question_score)

                    if single:
                        break
    if num_questions:
        avg_f1 = f1 / num_questions
    else:
        avg_f1 = INVALID_QUESTION
    return avg_f1, valid_questions, valid_cands, knowledge_answers, scores


def response_questions_stats(response, knowledge, gen_method, single, remove_personal, qg_model, qg_tokenizer, qa_model, qa_tokenizer):
    num_questions = 0
    num_no_ans = 0

    candidates = qg.get_answer_candidates(response)
    for cand in candidates:
        if gen_method == 'greedy':
            questions = [qg.get_question_greedy(cand, response, qg_model, qg_tokenizer)]
        elif gen_method == 'beam':
            questions = qg.get_questions_beam(cand, response, qg_model, qg_tokenizer)
        else:
            questions = qg.get_questions_sample(cand, response, qg_model, qg_tokenizer)

        for question in questions:
            if not remove_personal or non_personal(question):
                pred_ans = qa.get_answer(question, response, qa_model, qa_tokenizer)

                if filter_questions(cand, pred_ans) == 'VALID':
                    num_questions += 1
                    knowledge_ans = qa.get_answer(question, knowledge)
                    if knowledge_ans == NO_ANS:
                        num_no_ans += 1
                    if single:
                        break
    return num_questions, num_no_ans


def get_stats(in_path, gen_method, single, remove_personal, qa_model, qa_tokenizer):
    num_questions = 0
    num_no_ans = 0
    df = pd.read_csv(in_path)
    for _, row in df.iterrows():
        q, no_ans = response_questions_stats(row['response'], row['knowledge'], gen_method, single, remove_personal, qa_model, qa_tokenizer)
        num_questions += q
        num_no_ans += no_ans

    print("Total valid questions: {0}".format(num_questions))
    print("No answer: {0}".format(num_no_ans / num_questions))


def calc_scores(df, gen_method, single, remove_personal, save_steps=True, device=None):
    q_scores = []

    all_questions = []
    all_cands = []
    all_answers = []
    all_scores = []
    all_responses = []
    all_knowledge = []
    ids = []

    rows = []
    for idx, row in df.iterrows():
        rows.append(row)

    qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    qg_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap").to(device)

    qa_tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2").to(device)

    for idx, row in enumerate(tqdm(rows)):
        res, res_questions, res_cands, res_answers, res_scores =\
            get_response_score(row['response'], row['knowledge'], gen_method, single, remove_personal,
                               qg_model, qg_tokenizer, qa_model, qa_tokenizer)

        all_questions.extend(res_questions)
        all_cands.extend(res_cands)
        all_answers.extend(res_answers)
        all_scores.extend(res_scores)
        all_responses.extend([row['response']] * len(res_questions))
        all_knowledge.extend([row['knowledge']] * len(res_questions))
        ids.extend([idx] * len(res_questions))

        if res == INVALID_QUESTION:
            all_questions.extend([NO_VALID_QUESTIONS])
            all_cands.extend([NO_VALID_QUESTIONS])
            all_answers.extend([NO_VALID_QUESTIONS])
            all_scores.extend([INVALID_QUESTION])
            all_responses.extend([row['response'].lower()])
            all_knowledge.extend([row['knowledge']])
            ids.extend([idx])

        q_scores.append(res)

    df['Q2'] = q_scores
    df = df[df.Q2 >= 0]

    steps_df = None
    if save_steps:
        data = {'id': ids, 'response': all_responses, 'cand': all_cands, 'question': all_questions, 'knowledge': all_knowledge, 'knowledge_ans': all_answers, 'score': all_scores}
        steps_df = pd.DataFrame(data=data)

    valid_scores = [s for s in q_scores if s != -1]
    print("total with at least 1 valid question:", len(valid_scores))
    print("score:", np.mean(valid_scores))

    return df, steps_df

def q2_step_one(infile, gen_method="beam", q_per_cand="single", personal="remove", device=-1):
    if q_per_cand == 'single':
        single_q = True
    else:
        single_q = False

    if personal == 'remove':
        rm_personal = True
    else:
        rm_personal = False

    return calc_scores(infile, gen_method, single=single_q, remove_personal=rm_personal,
                 save_steps=True, device=device)
