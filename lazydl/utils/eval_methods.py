from lazydl.utils.result import Result
from lazydl.utils.log import Logger
from datasets import Dataset
import pandas as pd
from statistics import mean
import re
import string
from collections import Counter
from bert_score import score
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as meteor_scorer
from nltk.util import ngrams
from rouge import Rouge
from sacrebleu.metrics import BLEU, CHRF
import spacy
from rich.console import Console
from rich.table import Table
from typing import Union

log = Logger(__name__) 


class DistCritic:
    def __init__(self) -> None:
        self.name = "dist"
        
    def score_step(self, candidates: list[str], eval_result: Result, n_gram: int = 2, *args, **kwargs) -> Result:
        ngram_freqs = {}  
        ngram_len = 0  
        for candidate in candidates:
            for ngram in ngrams(word_tokenize(candidate), n_gram):
                ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
                ngram_len += 1
        # number of unique ngrams
        uniq_ngrams = len([val for val in ngram_freqs.values() if val == 1])
        distinct_ngram = len(ngram_freqs) / ngram_len if ngram_len > 0 else 0
        
        dist_score = round(distinct_ngram, 4)
        eval_result.merge({
            "DIST" + str(n_gram): dist_score
        })
        return eval_result

    def score(self, candidates: list[str], references: list[str], eval_result: Result, n_gram: int = 2, *args, **kwargs) -> Result:
        for step in range(n_gram):
            eval_result = self.score_step(candidates, eval_result, n_gram=step+1)
        
        return eval_result


class CorpusBleuCritic:
    def __init__(self) -> None:
        self.name = "corpus_bleu"

    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
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
        bleu1 = round(corpus_bleu(ref_list, dec_list, weights=(1, 0, 0, 0)) * 100, 4)
        bleu2 = round(corpus_bleu(ref_list, dec_list, weights=(0, 1, 0, 0)) * 100, 4)
        bleu3 = round(corpus_bleu(ref_list, dec_list, weights=(0, 0, 1, 0)) * 100, 4)
        bleu4 = round(corpus_bleu(ref_list, dec_list, weights=(0, 0, 0, 1)) * 100, 4)
        mean_corpus_bleu = round((bleu1 + bleu2 + bleu3 + bleu4) / 4, 4)
        
        eval_result.add(
            corpus_bleu1=bleu1,
            corpus_bleu2=bleu2,
            corpus_bleu3=bleu3,
            corpus_bleu4=bleu4,
            mean_corpus_bleu=mean_corpus_bleu,
        )
        
        return eval_result
    


class SentenceBleuCritic:
    def __init__(self) -> None:
        self.name = "sentence_bleu"

    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
        bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0
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
        bleu1 = round(bleu1 / len(ref_list)  * 100, 4)
        bleu2 = round(bleu2 / len(ref_list)  * 100, 4)
        bleu3 = round(bleu3 / len(ref_list)  * 100, 4)
        bleu4 = round(bleu4 / len(ref_list)  * 100, 4)
        mean_sent_bleu = round((bleu1 + bleu2 + bleu3 + bleu4) / 4, 4)
        
        eval_result.add(
            sent_bleu1=bleu1,
            sent_bleu2=bleu2,
            sent_bleu3=bleu3,
            sent_bleu4=bleu4,
            mean_sent_bleu=mean_sent_bleu,
        )
        
        return eval_result
    
    
class SacreSentenceBleuCritic:
    def __init__(self) -> None:
        self.name = "sacrebleu_sent"

    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
        bleu = BLEU()
        bleu_score = round(bleu.corpus_score(candidates, [references]).score, 4)
        eval_result.add(sacrebleu_sent=bleu_score)
        return eval_result
    
    
class SacreCorpusBleuCritic:
    def __init__(self) -> None:
        self.name = "sacrebleu_word"

    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
        bleu = BLEU()
        bleu_score = round(bleu.corpus_score(candidates, [references]).score, 4)
        eval_result.add(sacrebleu_corpus=bleu_score)
        return eval_result
    

class MeteorCritic:
    def __init__(self) -> None:
        self.name = "meteor"

    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
        score_list = []
        for i in range(len(candidates)):
            if type(references[i]) is list:
                ref_list = references[i]
            else:
                ref_list = [references[i]]
            ref = [r.split(" ") for r in ref_list]
            cand = candidates[i].split(" ")
            score = meteor_scorer(ref, cand)
            score_list.append(score)
        meteor_score = round(mean(score_list), 4)
        eval_result.add(Meteor=meteor_score)
        return eval_result

class RougeCritic:
    def __init__(self) -> None:
        self.name = "rouge"

    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
        rouge = Rouge()
        scores = rouge.get_scores(candidates, references)
        rouge_1 = round(mean([score["rouge-1"]["f"] * 100 for score in scores]), 4)
        rouge_2 = round(mean([score["rouge-2"]["f"] * 100 for score in scores]), 4)
        rouge_l = round(mean([score["rouge-l"]["f"] * 100 for score in scores]), 4)
        
        eval_result.add(
            Rouge_1=rouge_1,
            Rouge_2=rouge_2,
            Rouge_l=rouge_l,
        )
        
        return eval_result


class BertScoreCritic:
    """  "en": "roberta-large",
         "zh": "bert-base-chinese",
    """
    
    def __init__(self) -> None:
        self.name = "bertscore"
        

    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
        scores = score(
            candidates,
            references,
            lang="en",
            verbose=False,
            rescale_with_baseline=True,
            device="cuda:0",
        )[-1].numpy()
        bert_score = round(mean(list(scores)), 4)
        eval_result.add(bert_score=bert_score)
        
        return eval_result

    
class ChrfCritic:
    def __init__(self) -> None:
        self.name = "chrf"

    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
        chrf = CHRF(word_order=2)
        chrf_score = round(chrf.corpus_score(candidates, [references]).score, 4)
        eval_result.add(
            chrf_score=chrf_score
        )
        return eval_result
    
    

    
class F1NLPCritic:
    def __init__(self) -> None:
        self.name = "f1"
        self.nlp = None
        
    def set_nlp(self):
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        
    def get_tokens(self, text: str, nlp):
        doc = nlp(text)
        tokens = [tok.text.lower()
                for tok in doc if not tok.is_stop and not tok.is_punct]
        return tokens

    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
        """
        This function is copied from: https://github.com/orhonovich/q-squared/blob/main/pipeline/score.py
        """
        self.set_nlp()
        f1_list = []
        for i, a_gold in enumerate(references):
            a_pred = candidates[i]
            if a_pred == "":
                f1_list.append(0)
                continue
            gold_toks = self.get_tokens(a_gold, self.nlp)
            pred_toks = self.get_tokens(a_pred, self.nlp)
            common = Counter(gold_toks) & Counter(pred_toks)
            num_same = sum(common.values())
            if num_same == 0:
                f1_list.append(0)
                continue
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_list.append(f1)
        f1 = round(sum(f1_list) / len(f1_list) * 100, 2)
        
        eval_result.add(
            F1_NLP_split=f1
        )
        return eval_result
    
    
    
    
class F1SpaceCritic:
    def __init__(self) -> None:
        self.name = "f1"
        
        
    def clean_text(self, text: str):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the|in|our)\b", " ", text)
        return re.sub(" +", " ", text).strip()


    def score(self, candidates: list[str], references: list[str], eval_result: Result, *args, **kwargs) -> Result:
        """
        This function is copied from: https://github.com/orhonovich/q-squared/blob/main/pipeline/score.py
        """
        self.set_nlp()
        f1_list = []
        for i, a_gold in enumerate(references):
            a_pred = candidates[i]
            if a_pred == "":
                f1_list.append(0)
                continue
            gold_toks = self.clean_text(a_gold).split()
            pred_toks = self.clean_text(a_pred).split()
            common = Counter(gold_toks) & Counter(pred_toks)
            num_same = sum(common.values())
            if num_same == 0:
                f1_list.append(0)
                continue
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_list.append(f1)
        f1 = round(sum(f1_list) / len(f1_list) * 100, 2)
        
        eval_result.add(
            F1_Space_split=f1
        )
        return eval_result
    
    
class Q2Critic:
    def __init__(self) -> None:
        self.name = "q2"
        
    def set_env_pkg(self):
        from q2.run_pipeline import q2_step_one
        from q2.run_nli import q2_step_two
        self.q2_step_one = q2_step_one
        self.q2_step_two = q2_step_two
        
    def score(self, candidates: list[str], references: list[str], eval_result: Result, device: int = 0, eval_dataset: Dataset = None, *args, **kwargs) -> Result:
        self.set_env_pkg()
        df, steps_df = self.q2_step_one(infile=pd.DataFrame(eval_dataset)[["model_responses", "references", "q2_references"]].rename(columns={
            "model_responses": "response",
            "references": "gold",
            "q2_references": "knowledge"
        }), gen_method="beam", q_per_cand="single", personal="remove", device=device)
        q_squared_nli, q_squared_f1 = self.q2_step_two(infile=steps_df, device=device)
        
        eval_result.add(
            q_squared_nli=round(q_squared_nli, 4),
            q_squared_f1=round(q_squared_f1, 4)
        )
        
        return eval_result
    
    


Metrics = {
    "corpus_bleu": CorpusBleuCritic(),
    "sent_bleu": SentenceBleuCritic(),
    "sent_sacrebleu": SacreSentenceBleuCritic(),
    "corpus_sacrebleu": SacreCorpusBleuCritic(),
    "meteor": MeteorCritic(),
    "rouge": RougeCritic(),
    "bertscore": BertScoreCritic(),
    "chrf": ChrfCritic(),
    "dist": DistCritic(),
    "f1_nlp": F1NLPCritic(),
    "f1_space": F1SpaceCritic(),
}


def get_eval_metrics(eval_inputs: Union[Dataset, pd.DataFrame, dict] = None, 
                     metrics: list = [], 
                     device: int = 0):
    """计算评价指标，目前已支持的指标有: corpus_bleu, sent_bleu, sent_sacrebleu, corpus_sacrebleu, meteor, rouge, bertscore, chrf, dist, f1_nlp, f1_space
    

    Args:
        eval_inputs (Union[Dataset, pd.DataFrame, dict], optional): _description_. Defaults to None.
        metrics (list, optional): _description_. Defaults to [].
        device (int, optional): _description_. Defaults to 0.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if not isinstance(eval_inputs, Dataset):
        if isinstance(eval_inputs, pd.DataFrame):
            eval_inputs = Dataset.from_pandas(eval_inputs)
        elif isinstance(eval_inputs, dict):
            eval_inputs = Dataset.from_dict(eval_inputs)
        else:
            raise ValueError("评价指标计算的输入 eval_inputs 必须是 Dataset, pd.DataFrame 或者 dict类型，并满足转为 Dataset 的条件。")
    eval_result = Result()
    if "model_responses" in eval_inputs.column_names:
        model_responses = eval_inputs["model_responses"]
    else:
        model_responses = eval_inputs["generated"]
    if "references" in eval_inputs.column_names:
        references = eval_inputs["references"]
    else:
        return eval_result
    
    if "" in model_responses:
        log.warning("评价指标计算的输入中（model_responses）包含空值，评测结果或有偏颇。")
        
    
    for metric in metrics:
        if metric not in Metrics:
            log.warning("评价指标 {} 不在可用的评价指标列表中，将跳过该指标。".format(metric))
            continue
        
        log.info("计算 {} 评价指标...".format(metric))
        
        try:
            eval_result = Metrics[metric].score(candidates=model_responses, references=references, eval_result=eval_result, device=device, eval_dataset=eval_inputs)
        except Exception as e:
            log.error(e)
            log.warning("评价指标 {} 计算失败，将跳过该指标。".format(metric))

    console = Console(color_system="256", style="cyan")
    table = Table(style="cyan", show_footer=False, title="[bold green]Evaluation results")
    table.add_column("Metric", justify="right", style="magenta")
    table.add_column("Score :yum:", justify="left", style="magenta")
    for k, v in eval_result.items():
        table.add_row(k, str(v))
    console.print(table)
    
    return eval_result
    




