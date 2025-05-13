maximums = []

maximums.append([0] * 12) # 12; max < 50
maximums.append([0]) # max < 50
maximums.append([0] * 4) # 4; max < 300
maximums.append([0]) # max < 300
maximums.append([0]) # max < 2000
maximums.append([0]) # max < 8
maximums.append([0]) # max < 8
maximums.append([0]) # max < 50
maximums.append([0]) # max < 10
for molecule in tqdm(data['descriptors']):
    for substr in molecule:
        if substr == '$':
            continue
        for i, descriptor in enumerate(substr):
            for j, number in enumerate(descriptor):
                if number == 'Invalid':
                    continue
                maximums[i][j] = max(number, maximums[i][j])
maximums