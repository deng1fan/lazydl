def rename_columns(dataset, column_map):
    for old_name, new_name in column_map.items():
        dataset = dataset.rename_column(old_name, new_name)
    return dataset