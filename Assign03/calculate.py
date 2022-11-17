import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test = 'Michalewicz'
dim_list = [2, 10]
print(f'----------{test}----------')
for dim in dim_list:
    print(f'-----{dim}-----')
    pop_array = [32, 64, 128, 256, 512, 1024]

    for pop in pop_array:
        
        print(f'- {pop}')
        print("-- 1. DE")
        all_fitness_DE = []    
        for path in sorted(os.listdir(f'log/DE/{dim}/{test}/{pop}/')):
            df = pd.read_csv(f'log/DE/{dim}/{test}/{pop}/' + path)
            all_fitness_DE.append(df['Fitness'].min())
        print(np.mean(all_fitness_DE))
        print(np.std(all_fitness_DE))
        
        print('--2. CEMv2')
        all_fitness_CEMv2 = []    
        for path in sorted(os.listdir(f'log/CEMv2/{dim}/{test}/{pop}/')):
            df = pd.read_csv(f'log/CEMv2/{dim}/{test}/{pop}/' + path)
            all_fitness_CEMv2.append(df['Fitness'].min())
        print(np.mean(all_fitness_CEMv2))
        print(np.std(all_fitness_CEMv2))