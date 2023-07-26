def tokenize(tokenizer, examples, columns_map, *args, **kwargs):
    for new_name, old_name in columns_map.items():
        if type(examples[old_name][0]) != str:
            examples[old_name] = [str(x) for x in examples[old_name]]
        examples[new_name] = tokenizer(examples[old_name], *args, **kwargs)["input_ids"]
    return examples


def advance_tokenize(tokenizer, dataset, columns_map, *args, **kwargs):
    dataset = dataset.map(
        lambda examples: tokenize(
            tokenizer, examples, columns_map, *args, **kwargs
        ),
        batched=True,
        num_proc=4,
        desc="Running tokenizer on dataset",
    )
    return dataset
    