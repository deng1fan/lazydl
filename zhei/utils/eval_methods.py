from zhei.utils.result import Result
from zhei.utils.log import Logger
from zhei.utils.catch_error import print_error_info
from datasets import Dataset
import pandas as pd
from evaluate import load
from statistics import mean
import re
import string
from collections import Counter
from bert_score import score
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.util import ngrams
from rouge import Rouge
from sacrebleu.metrics import BLEU, CHRF
import sacrebleu
import spacy
from rich.console import Console
from rich.table import Table
from typing import Union

log = Logger(__name__) 

def distinct_ngram(candidates, n=2):
    """Return basic ngram statistics, as well as a dict of all ngrams and their freqsuencies."""
    ngram_freqs = {}  # ngrams with frequencies
    ngram_len = 0  # total number of ngrams
    for candidate in candidates:
        for ngram in ngrams(word_tokenize(candidate), n):
            ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
            ngram_len += 1
    # number of unique ngrams
    uniq_ngrams = len([val for val in ngram_freqs.values() if val == 1])
    distinct_ngram = len(ngram_freqs) / ngram_len if ngram_len > 0 else 0
    return round(distinct_ngram, 4)

def compute_meteor(references, candidates):
    score_list = []
    for i in range(len(candidates)):
        if type(references[i]) is list:
            ref_list = references[i]
        else:
            ref_list = [references[i]]
        ref = [r.split(" ") for r in ref_list]
        cand = candidates[i].split(" ")
        score = meteor_score(ref, cand)
        score_list.append(score)
    return round(mean(score_list), 4)

def compute_sent_bleu(references, candidates):
    bleu1 = 0.0
    bleu2 = 0.0
    bleu3 = 0.0
    bleu4 = 0.0
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(word_tokenize(candidates[i]))
        if type(references[i]) is list:
            tmp = []
            for ref in references[i]:
                tmp.append(word_tokenize(ref))
            ref_list.append(tmp)
        else:
            ref_list.append([word_tokenize(references[i])])

    for example_id, (label, pred) in enumerate(zip(ref_list, dec_list)):
        bleu1 += sentence_bleu(
            label,
            pred,
            weights=[1, 0, 0, 0],
            smoothing_function=SmoothingFunction().method3,
        )
        bleu2 += sentence_bleu(
            label,
            pred,
            weights=[0.5, 0.5, 0, 0],
            smoothing_function=SmoothingFunction().method3,
        )
        bleu3 += sentence_bleu(
            label,
            pred,
            weights=[1 / 3, 1 / 3, 1 / 3, 0],
            smoothing_function=SmoothingFunction().method3,
        )
        bleu4 += sentence_bleu(
            label,
            pred,
            weights=[0.25, 0.25, 0.25, 0.25],
            smoothing_function=SmoothingFunction().method3,
        )
    bleu1 = bleu1 / len(ref_list)
    bleu2 = bleu2 / len(ref_list)
    bleu3 = bleu3 / len(ref_list)
    bleu4 = bleu4 / len(ref_list)
    return (
        round(bleu1 * 100, 4),
        round(bleu2 * 100, 4),
        round(bleu3 * 100, 4),
        round(bleu4 * 100, 4),
    )
    
def compute_rouge(references, candidates):
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references)
    rouge_1 = [score["rouge-1"]["f"] * 100 for score in scores]
    rouge_2 = [score["rouge-2"]["f"] * 100 for score in scores]
    rouge_l = [score["rouge-l"]["f"] * 100 for score in scores]
    return (
        round(mean(rouge_1), 5),
        round(mean(rouge_2), 5),
        round(mean(rouge_l), 5),
    )

def compute_chrf(references, candidates):
    chrf = CHRF(word_order=2)
    return round(chrf.corpus_score(candidates, [references]).score, 4)


def compute_sacre_corpus_bleu(references, candidates):
    bleu = BLEU()
    return round(bleu.corpus_score(candidates, [references]).score, 4)


def compute_sacre_sent_bleu(references, candidates):
    bleu = []
    for i, a_gold in enumerate(references):
        bleu.append(sacrebleu.corpus_bleu([candidates[i]], [[a_gold]]).score)
    return round(mean(bleu), 4)

def get_tokens(text, nlp):
    doc = nlp(text)
    tokens = [tok.text.lower()
              for tok in doc if not tok.is_stop and not tok.is_punct]
    return tokens


def get_bert_score(references, candidates, device):
    # scorer = BertScorer()
    # scorer.init_scorer(lang='en', num_layers=8, rescale_with_baseline=True)
    # scores = scorer.get_score(target, generated)[-1]
    scores = score(
        candidates,
        references,
        lang="en",
        verbose=False,
        rescale_with_baseline=True,
        device=device,
    )[-1].numpy()
    return round(mean(list(scores)), 4)

def compute_corpus_bleu(references, candidates):
    bleu1 = 0.0
    bleu2 = 0.0
    bleu3 = 0.0
    bleu4 = 0.0
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(word_tokenize(candidates[i]))
        if type(references[i]) is list:
            tmp = []
            for ref in references[i]:
                tmp.append(word_tokenize(ref))
            ref_list.append(tmp)
        else:
            ref_list.append([word_tokenize(references[i])])
    bleu1 = corpus_bleu(ref_list, dec_list, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(ref_list, dec_list, weights=(0, 1, 0, 0))
    bleu3 = corpus_bleu(ref_list, dec_list, weights=(0, 0, 1, 0))
    bleu4 = corpus_bleu(ref_list, dec_list, weights=(0, 0, 0, 1))
    return (
        round(bleu1 * 100, 4),
        round(bleu2 * 100, 4),
        round(bleu3 * 100, 4),
        round(bleu4 * 100, 4),
    )

def compute_f1(references, candidates, nlp=None):
    """
    This function is copied from: https://github.com/orhonovich/q-squared/blob/main/pipeline/score.py
    """
    f1_list = []
    for i, a_gold in enumerate(references):
        a_pred = candidates[i]
        if a_pred == "":
            f1_list.append(0)
            continue
        if nlp:
            gold_toks = get_tokens(a_gold, nlp)
            pred_toks = get_tokens(a_pred, nlp)
        else:
            gold_toks = clean_text(a_gold).split()
            pred_toks = clean_text(a_pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if num_same == 0:
            f1_list.append(0)
            continue
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_list.append(f1)
    return round(sum(f1_list) / len(f1_list) * 100, 2)


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the|in|our)\b", " ", text)
    return re.sub(" +", " ", text).strip()




def get_eval_metrics(outputs: Union[Dataset, pd.DataFrame, dict] = None, 
                     metrics: list = [], 
                     device: int = 0):
    """计算评价指标

    Args:
        outputs (Union[Dataset, pd.DataFrame, dict], optional): 模型的输出
        metrics (list, optional): 需要评价的指标. Defaults to [].
        device (int, optional): 如果涉及 GPU 计算，想要使用的 GPU 序号，应该为单 GPU. Defaults to 0.
        
    """
    if not isinstance(outputs, Dataset):
        try:
            if isinstance(outputs, pd.DataFrame):
                outputs = Dataset.from_pandas(outputs)
            elif isinstance(outputs, dict):
                outputs = Dataset.from_dict(outputs)
            else:
                raise ValueError()
        except Exception as e:
            raise ValueError("评价指标计算的输入必须是 Dataset, pd.DataFrame 或者 dict类型")
    test_result = Result()
    if "generated_seqs" in outputs.column_names:
        generated_seqs = outputs["generated_seqs"]
    else:
        generated_seqs = outputs["generated"]
    if "reference" in outputs.column_names:
        reference = outputs["reference"]
    else:
        return test_result

    # ---------------------------------------------------------------------------- #
    #                         计算 Meteor                                     
    # ---------------------------------------------------------------------------- #
    if "meteor" in metrics:
        log.info("计算 Meteor ing...")
        meteor_score = compute_meteor(reference, generated_seqs)
        test_result.add(meteor=meteor_score)
        log.info(f"Meteor = {str(meteor_score)}")

    # ---------------------------------------------------------------------------- #
    #                         计算 chrf                                     
    # ---------------------------------------------------------------------------- #
    if "hf_chrf" in metrics:
        log.info("计算 chrF ing...")
        try:
            chrf_computer = load("chrf")
            chrf = chrf_computer.compute(
                predictions=generated_seqs, references=reference)['score']
            hf_chrf = round(chrf, 4)
        except Exception as e:
            log.error("计算 hf_chrf 失败")
            print_error_info(e)
            hf_chrf = 0
        test_result.add(chrf=hf_chrf)
        log.info(f"chrf = {str(hf_chrf)}")

    if "chrf" in metrics:
        try:
            chrf = compute_chrf(reference, generated_seqs)  # 偏低3个点
        except Exception as e:
            log.error("计算 chrf 失败")
            print_error_info(e)
            chrf = 0
        test_result.add(chrf=chrf)
        log.info(f"chrf = {str(chrf)}")

    # ---------------------------------------------------------------------------- #
    #                                  计算F1                                     
    #     https://github.com/orhonovich/q-squared/blob/main/pipeline/score.py
    # ---------------------------------------------------------------------------- #
    if "f1_space_split" in metrics:
        log.info("计算 f1_space_split ing...")
        f1_reference = outputs["f1_reference"]
        f1 = compute_f1(f1_reference, generated_seqs)
        test_result.add(f1_space_split=f1)
        log.info(f"f1_space_split = {str(f1)}")

    if "f1_nlp_split" in metrics:
        log.info("计算 f1_nlp_split ing...")
        nlp = spacy.load("en_core_web_sm")
        f1_reference = outputs["f1_reference"]
        f1 = compute_f1(f1_reference, generated_seqs, nlp)
        test_result.add(f1_nlp_split=f1)
        log.info(f"f1_nlp_split = {str(f1)}")

    # ---------------------------------------------------------------------------- #
    #                         计算 google_bleu                                     
    # ---------------------------------------------------------------------------- #
    if "hf_google_bleu" in metrics:
        log.info("计算 hf_google_bleu ing...")
        try:
            google_bleu = load("hf_google_bleu")
            google_bleu_score = google_bleu.compute(
                predictions=generated_seqs, references=reference
            )["hf_google_bleu"]
            google_bleu_score = google_bleu_score * 100
        except Exception as e:
            log.error("计算 hf_google_bleu 失败")
            print_error_info(e)
            google_bleu_score = 9999
        google_bleu_score = round(google_bleu_score, 4)
        test_result.add(google_bleu=google_bleu_score)
        log.info(f"hf_google_bleu = {str(google_bleu_score)}")

    # ---------------------------------------------------------------------------- #
    #                         计算 sacrebleu                                     
    # ---------------------------------------------------------------------------- #
    if "hf_sacrebleu" in metrics:
        log.info("计算 HuggingFcae Sacre BLEU ing...")
        try:
            sacrebleu_computer = load("sacrebleu")
            bleu = sacrebleu_computer.compute(
                predictions=generated_seqs, references=reference)["score"]
            bleu = round(bleu, 4)
        except Exception as e:
            log.error("计算 hf_sacrebleu 失败")
            print_error_info(e)
            bleu = 0
        test_result.add(sacrebleu_huggingface=bleu)
        log.info(f"hf_sacrebleu = {str(bleu)}")

    if "sacrebleu_sent" in metrics:
        log.info("计算 Sacre BLEU sent level ing...")
        try:
            bleu = compute_sacre_sent_bleu(
                candidates=generated_seqs, references=reference)
        except Exception as e:
            log.error("计算 sacrebleu_sent 失败")
            print_error_info(e)
            bleu = 0
        test_result.add(sacrebleu_sent=bleu)
        log.info(f"sacrebleu_sent = {str(bleu)}")

    if "sacrebleu_corpus" in metrics:
        log.info("计算 Sacre BLEU corpus level ing...")
        try:
            bleu = compute_sacre_corpus_bleu(
                candidates=generated_seqs, references=reference)
        except Exception as e:
            log.error("计算 sacrebleu_corpus 失败")
            print_error_info(e)
            bleu = 0
        test_result.add(sacrebleu_corpus=bleu)
        log.info(f"sacrebleu_corpus = {str(bleu)}")

    # ---------------------------------------------------------------------------- #
    #                         sent_bleu                                     
    # ---------------------------------------------------------------------------- #
    if "sent_bleu" in metrics:
        bleu1, bleu2, bleu3, bleu4 = compute_sent_bleu(
            reference, generated_seqs)
        mean_sent_bleu = round((bleu1 + bleu2 + bleu3 + bleu4) / 4, 4)
        test_result.add(
            sent_bleu1=bleu1,
            sent_bleu2=bleu2,
            sent_bleu3=bleu3,
            sent_bleu4=bleu4,
            mean_sent_bleu=mean_sent_bleu,
        )
        log.info(f"sent_bleu1 = {str(bleu1)}")
        log.info(f"sent_bleu2 = {str(bleu2)}")
        log.info(f"sent_bleu3 = {str(bleu3)}")
        log.info(f"sent_bleu4 = {str(bleu4)}")
        log.info(f"mean_sent_bleu = {str(mean_sent_bleu)}")

    # ---------------------------------------------------------------------------- #
    #                         corpus_bleu                                     
    # ---------------------------------------------------------------------------- #
    if "corpus_bleu" in metrics:
        bleu1, bleu2, bleu3, bleu4 = compute_corpus_bleu(
            reference, generated_seqs)
        mean_corpus_bleu = round((bleu1 + bleu2 + bleu3 + bleu4) / 4, 4)
        test_result.add(
            corpus_bleu1=bleu1,
            corpus_bleu2=bleu2,
            corpus_bleu3=bleu3,
            corpus_bleu4=bleu4,
            mean_corpus_bleu=mean_corpus_bleu,
        )
        log.info(f"corpus_bleu1 = {str(bleu1)}")
        log.info(f"corpus_bleu2 = {str(bleu2)}")
        log.info(f"corpus_bleu3 = {str(bleu3)}")
        log.info(f"corpus_bleu4 = {str(bleu4)}")
        log.info(f"mean_corpus_bleu = {str(mean_corpus_bleu)}")

    # ---------------------------------------------------------------------------- #
    #                         计算 Dist                                     
    # ---------------------------------------------------------------------------- #
    if "dist" in metrics:
        log.info("计算 Dist ing...")
        dist1 = distinct_ngram(generated_seqs, n=1)
        dist2 = distinct_ngram(generated_seqs, n=2)
        test_result.add(
            dist1=dist1,
            dist2=dist2,
        )
        log.info(f"dist1 = {str(dist1)}")
        log.info(f"dist2 = {str(dist2)}")

    # ---------------------------------------------------------------------------- #
    #                         计算 ROUGE                                     
    # ---------------------------------------------------------------------------- #
    if "hf_rouge" in metrics:
        log.info("计算 hf_rouge ing...")
        try:
            rouge = load('rouge')

            rouge_results = rouge.compute(predictions=generated_seqs,
                                          references=reference)
            rouge_1 = round(rouge_results['rouge1'], 5)
            rouge_2 = round(rouge_results['rouge2'], 5)
            rouge_L = round(rouge_results['rougeL'], 5)
            rouge_Lsum = round(rouge_results['rougeLsum'], 5)
            test_result.add(rouge_1=rouge_1)
            test_result.add(rouge_2=rouge_2)
            test_result.add(rouge_L=rouge_L)
            test_result.add(rouge_Lsum=rouge_Lsum)
            log.info(f"hf_rouge_1 = {str(rouge_1)}")
            log.info(f"hf_hf_rouge_2 = {str(rouge_2)}")
            log.info(f"hf_rouge_L = {str(rouge_L)}")
            log.info(f"hf_rouge_Lsum = {str(rouge_Lsum)}")

        except Exception as e:
            print_error_info(e)
            log.info("hf_rouge 无法计算，Reference可能为空，请检查生成数据！")

    if "rouge" in metrics:
        log.info("计算 ROUGE ing...")
        try:
            rouge_1, rouge_2, rouge_l = compute_rouge(reference, generated_seqs)
            test_result.add(rouge_1=rouge_1)
            test_result.add(rouge_2=rouge_2)
            test_result.add(rouge_l=rouge_l)
            log.info(f"rouge_1 = {str(rouge_1)}")
            log.info(f"rouge_2 = {str(rouge_2)}")
            log.info(f"rouge_L = {str(rouge_l)}")

        except Exception as e:
            print_error_info(e)
            log.info("Rouge 无法计算，Reference可能为空，请检查生成数据！")

    # ---------------------------------------------------------------------------- #
    #                         计算 Bert Score                                     
    # ---------------------------------------------------------------------------- #
    if "hf_bert_score" in metrics:
        log.info("计算 hf_bert_score ing...")
        bert_score_reference = outputs["bert_score_reference"]
        try:
            bertscore = load("bertscore")
            bert_score = mean(bertscore.compute(predictions=generated_seqs,
                                                references=bert_score_reference, lang="en", rescale_with_baseline=True)['f1'])
            bert_score = round(bert_score, 4)
            test_result.add(bert_score=bert_score)
            log.info(f"hf_bert_score = {str(bert_score)}")
        except ValueError:
            log.info("hf_bert_score 无法计算，Reference可能为空，请检查生成数据！")

    if "bert_score" in metrics:
        log.info("计算 Bert score ing...")
        bert_score_reference = outputs["bert_score_reference"]
        try:
            bert_score = get_bert_score(bert_score_reference, generated_seqs, device)
            test_result.add(bert_score=bert_score)
            log.info(f"bert_score = {str(bert_score)}")
        except ValueError:
            log.info("bert_score 无法计算，Reference可能为空，请检查生成数据！")

    # ---------------------------------------------------------------------------- #
    #                         计算 PPL                                     
    # ---------------------------------------------------------------------------- #
    if "hf_ppl" in metrics:
        log.info("计算 hf_ppl ing...")
        perplexity = load("perplexity", module_type="metric")
        try:
            ppl = perplexity.compute(predictions=generated_seqs, model_id="gpt2")[
                "mean_perplexity"
            ]
        except Exception as e:
            log.error("计算 PPL 失败")
            print_error_info(e)
            ppl = 9999
        ppl = round(ppl, 4)
        test_result.add(ppl=ppl)
        log.info(f"hf_ppl = {str(ppl)}")

    # ---------------------------------------------------------------------------- #
    #                         计算 q_squared                                     
    # ---------------------------------------------------------------------------- #
    if "q_squared" in metrics:
        from q2.run_pipeline import q2_step_one
        from q2.run_nli import q2_step_two
        log.info("计算 q_squared ing...")

        try:
            df, steps_df = q2_step_one(infile=pd.DataFrame(outputs)[["generated_seqs", "reference", "q2_reference"]].rename(columns={
                "generated_seqs": "response",
                "reference": "gold",
                "q2_reference": "knowledge"
            }), gen_method="beam", q_per_cand="single", personal="remove", device=device)
            q_squared_nli, q_squared_f1 = q2_step_two(infile=steps_df, device=device)
            test_result.add(q_squared_nli=round(q_squared_nli, 4))
            test_result.add(q_squared_f1=round(q_squared_f1, 4))
            log.info(f"q_squared_nli = {str(q_squared_nli)}")
            log.info(f"q_squared_f1 = {str(q_squared_f1)}")
        except Exception as e:
            print_error_info(e)
            log.info("q_squared 无法计算，请检查数据！")

    console = Console(color_system="256", style="cyan")
    table = Table(style="cyan", show_footer=False, title="[bold green]Evaluation results")
    table.add_column("Metric", justify="right", style="magenta")
    table.add_column("Score:yum:", justify="left", style="magenta")
    for k, v in test_result.items():
        table.add_row(k, str(v))
    console.print(table)
    return test_result