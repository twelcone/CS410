import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test_function import Rosenbrock, Sphere, Ackley, Zakharov, Michalewicz
from constant import seed_number, F_scale, cross_prob
from constant import sigma_init, dimension
from constant import num_elite, sigma_init, extra_std
from algorithm import CEMv2
from celluloid import Camera
    

def CrossEntropyMethodv2(func_name, l, r, x_mark , y_mark, eps,  test_function, dimensions, bounds, popsize, num_elite, sigma_init, extra_std, seed_number, max_evals):
    np.random.rand(seed_number)
    results = []
    cov_matrix = []
    
    max_evals = 1e5 if dimensions == 2 else 1e6
    
    all_pops, cov_matrix, results, generation_count = CEMv2(test_function, dimensions, eps, bounds, popsize, num_elite, sigma_init, extra_std, seed_number, max_evals)

    bound_lower = l
    bound_upper = r
    x = np.linspace(bound_lower, bound_upper, 100)
    y = np.linspace(bound_lower, bound_upper, 100)
    X, Y = np.meshgrid(x, y)
    Z = test_function([X, Y])
    
    fig = plt.figure(figsize=(12, 12))
    camera = Camera(fig)
    plt.contourf(X, Y, Z, popsize, cmap='viridis')
    plt.axis('square')
    plt.scatter(x_mark, y_mark, marker='*')
    
    if popsize == 32 and dimensions == 2:
        for generation in range(len(all_pops)):
            plt.contourf(X, Y, Z, popsize, cmap='viridis')
            plt.axis('square')
            plt.scatter(x_mark, y_mark, marker='*')
            plt.scatter(all_pops[generation][:, 0], all_pops[generation][:, 1], c='#ff0000', marker='o')
            plt.scatter(cov_matrix[generation][:, 0], cov_matrix[generation][:, 1], c='#ff1493', marker='x')
            plt.plot()
            plt.xlim((bound_lower, bound_upper))
            plt.ylim((bound_lower, bound_upper))
            # plt.pause(0.1)
            camera.snap()
        # plt.show()
        anim = camera.animate()
        
        #save the animation as a gif file
        anim.save('gif/CEMv2/' + func_name + "-CEMv2-" + str(popsize) + str(seed_number)+ ".gif",writer="pillow")
        plt.close()
        del anim, camera, fig
        # gif = False

    # if not os.path.exists(f'log/CEMv2/{str(dimensions)}/{func_name}/{str(popsize)}'): 
    #     os.mkdir(f'log/CEMv2/{str(dimensions)}/{func_name}/{str(popsize)}')
    # df = pd.DataFrame(results, columns=['Best Ind', 'Fitness', '#Evals'])
    # df.index.name = '#Gen'
    # df.to_csv(f'log/CEMv2/{str(dimensions)}/{func_name}/{str(popsize)}/{seed_number}.csv')

def choose_func(func):
    if func == "Sphere": return Sphere, 0.000001, -5.12, 5.12, 0, 0
    elif func == "Rosenbrock": return Rosenbrock, 0.000001, -5, 10, 0, 0
    elif func == "Ackley": return Ackley, 0.000001, -32.768, 32.768, 0, 0
    elif func == "Zakharov": return Zakharov, 0.000001, -5, 10, 0, 0
    elif func == "Michalewicz": return Michalewicz, -1.8013, 0, np.pi, 2.2, 1.57

if __name__=='__main__':
    all_fitness = []
    num_evaluation = []
    
    seed_number = 19520448
    # "Sphere", "Rosenbrock", "Ackley", "Michalewicz"
    func_name_list = ["Sphere"]
    for func_name in func_name_list:
        gif = True
        print(f"FUNCTION: {func_name}")
        if func_name == "Michalewicz": 
            eps = -1.8013
            # eps = -9.66015 
        else: eps = 1e-4
        popsize_array = [32]
        for popsize in popsize_array:
            print(f"Popsize = {popsize}")
            seed_number = 19520448
            if dimension == 2:
                max_evals = 1e5
            else: max_evals = 1e6
            
            print(f"- {seed_number}")
            
            test_function, eps , bound_lower, bound_upper, x_mark, y_mark = choose_func(func_name)
            bounds = [(bound_lower, bound_upper)]*dimension
            
            CrossEntropyMethodv2(func_name, bound_lower, bound_upper, x_mark, y_mark, eps, test_function, dimension, bounds, popsize, num_elite, sigma_init, extra_std, seed_number, max_evals) 

        
