START_TOKEN = 0
PADDING_TOKEN = 1
END_TOKEN = 2
SPLITTER_TOKEN = 3
DOLLAR_TOKEN = 4

def append_everything_and_add_special_tokens(descriptors_of_substructures, max_length, truncation=True):
    mask = [1] * max_length
    susbstr_merged_all_appended = []
    susbstr_merged_all_appended.append(START_TOKEN)
    
    for i, substructure in enumerate(descriptors_of_substructures):
        if substructure == '$':
            susbstr_merged_all_appended.append(DOLLAR_TOKEN)
            continue
        
        for descriptor in substructure:
            for number in descriptor:
                susbstr_merged_all_appended.append(number)
        
        if i != len(descriptors_of_substructures) - 1:
            susbstr_merged_all_appended.append(SPLITTER_TOKEN)
            
    susbstr_merged_all_appended.append(END_TOKEN)
    
    if len(susbstr_merged_all_appended) > max_length:
        susbstr_merged_all_appended = susbstr_merged_all_appended[:max_length]
        
    real_len = len(susbstr_merged_all_appended)
    for i in range(real_len, max_length):
        mask[i] = 0
        
    while len(susbstr_merged_all_appended) < max_length:
        susbstr_merged_all_appended.append(PADDING_TOKEN)
    
    return susbstr_merged_all_appended, mask

def tokenize(array_of_descriptors_of_substructures, max_length):
    all_appended_array = []
    mask_array = []
    for descriptors_of_substructures in array_of_descriptors_of_substructures:
        all_appended, mask = append_everything_and_add_special_tokens(descriptors_of_substructures, max_length)
        all_appended_array.append(all_appended)
        mask_array.append(mask)
    sample = {
        'input_ids': all_appended_array,
        'attention_mask': mask_array
    }
    return sample
    