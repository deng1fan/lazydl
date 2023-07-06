from datasets import load_dataset


def load_hf_faithdial(*args, **kwargs):
    """加载 FaithDial 数据集（英文），对话领域，对 WoW 数据集中的幻觉进行了修正
    
    train: 36809 turns
    valid: 6851 turns
    test: 7101 turns
    
    [
        {
            "utterances": [
            ... // prior utterances, 
            {
                "history": [
                "Have you ever been to a concert? They're so fun!",
                ],
                "speaker": "Wizard",
                "knowledge": "It began on September 9, 2015, in Montreal, Canada, at the Bell Centre and concluded on March 20, 2016, in Sydney, Australia at Allphones Arena.",
                "original_response": "It started in September of 2015 and ran all the way through March of 2016. Can you imagine being on the road that long?",
                "response": "Sure. The concert started in September 9th of 2015 at Montreal, Canada. It continued till 20th of March of 2016, where it ended at Sydney, Australia.",
                "BEGIN": [
                "Hallucination",
                "Entailment"
                ],
                "VRM": [
                "Disclosure",
                "Question"
                ]
            }, 
            ... // more utterances
            ]
        }, 
        ... // more dialogues
    ]
    
    """
    return load_dataset("McGill-NLP/FaithDial", *args, **kwargs)


def load_hf_wizard_of_wikipedia(*args, **kwargs):
    """加载 WoW 数据集（英文），对话领域, 一个对话者随机选择一个初始话题，对话双方可以在此基础上进行对话，但在对话过程中话题也可以拓展。对话双方的角色是不同的，分为 wizard 和 apprentice：

        wizard：wizard 的目的是通知 apprentice 关于对话主题相关的背景知识，在对话开始之前，会给定一些相关的 wiki 段落，这些对于 apprentice 不可见。同时，wizard 不允许直接复制拷贝 wiki 里的文本句子作为回复，而是需要自己进行组合生成融合知识的回答。
        apprentice：apprentice 的目的是深入的询问与对话主题相关的问题，这与普通的闲聊有所区别。
    
    测试集分为 seen 和 unseen 两种，seen 测试集中的话题在训练集中出现过，unseen 测试集中的话题在训练集中没有出现过。
    
    整个数据集超过 2W 个对话，540W 篇文档

    """
    return load_dataset("chujiezheng/wizard_of_wikipedia", *args, **kwargs)


def load_hf_CEval(*args, **kargs):
    """加载 C-Eval 数据集（中文），C-Eval是全面的中文基础模型评估套件，涵盖了52个不同学科的13948个多项选择题，分为四个难度级别
    
    Github: https://github.com/SJTU-LIT/ceval/tree/main

    Returns:
        _type_: _description_
    """
    return load_dataset("ceval/ceval-exam", *args, **kargs)


def load_AGIEval(dir: str = None, prompt_path: str = None, *args, **kargs):
    """加载 A-GIEval 数据集（中文），A-GIEval是全面的中文基础模型评估套件，涵盖了52个不同学科的13948个多项选择题，分为四个难度级别
    
    Github: https://github.com/microsoft/AGIEval
    prompt_data_csv: https://github.com/microsoft/AGIEval/blob/main/data/few_shot_prompts.csv
    
    {
        "passage": null,
        "question": "设集合 $A=\\{x \\mid x \\geq 1\\}, B=\\{x \\mid-1<x<2\\}$, 则 $A \\cap B=$ ($\\quad$)\\\\\n",
        "options": ["(A)$\\{x \\mid x>-1\\}$", 
            "(B)$\\{x \\mid x \\geq 1\\}$", 
            "(C)$\\{x \\mid-1<x<1\\}$", 
            "(D)$\\{x \\mid 1 \\leq x<2\\}$"
            ],
        "label": "D",
        "answer": null
    }
    
    """
    print("AGIEval 数据集正在开发中...")
    from zhei.utils.data_utils.AGIEval import post_process, utils, dataset_loader
    dataset_name_list = [
        "aqua-rat",
        "math",
        "logiqa-en", "logiqa-zh",
         "jec-qa-kd", "jec-qa-ca",
        "lsat-ar", "lsat-lr", "lsat-rc",
        "sat-math", "sat-en",
        "sat-en-without-passage",
        "gaokao-chinese",
        "gaokao-english",
        "gaokao-geography", "gaokao-history",
        "gaokao-biology", "gaokao-chemistry", "gaokao-physics",
        "gaokao-mathqa",
        "gaokao-mathcloze",
    ]
    setting_name_list = [
        'zero-shot',
        'zero-shot-CoT',
        'few-shot',
        'few-shot-CoT',
    ]
    datasets = {}
    for dataset_name in dataset_name_list:
        for setting_name in setting_name_list:
            if 'few-shot' in setting_name:
                continue
            dataset = dataset_loader.load_dataset(
                dataset_name, setting_name, dir,
                prompt_path=prompt_path, max_tokens=2048,
                end_of_example="<END>\n", verbose=True)
            datasets[f"{dataset_name}-{setting_name}"] = dataset
            
    return datasets
    