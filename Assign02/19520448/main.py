import numpy as np
import time 

def initialize_population( num_individuals, num_variables ): 
    """ 
    Initialize population gồm num_individuals cá thể. Mỗi cá thể có num_parameters biến.
    
    Arguments:
    num_individuals -- Số lượng cá thể
    num_variables -- Số lượng biến
    
    Returns:
    pop -- Ma trận (num_individuals, num_variables) chứa quần thể mới được khởi tạo ngẫu nhiên.
    """
    pop = np.random.randint(2, size=(num_individuals, num_variables)) 
    return pop

def initialize_random_seed( mssv ): 
    """ 
        Hàm tạo bộ random_seed
    """  
    random_seed = [(mssv + i) for i in range(0, 100)]
    random_seed = np.reshape(random_seed, (10, 10))
    return random_seed

def onemax( ind ):   
    """
    Hàm đánh giá OneMax: Đếm số bit 1 trong chuỗi nhị phân (cá thể ind).
    
    Arguments:
    ind -- Cá thể cần được đánh giá.

    Returns:
    value -- Giá trị của cá thể ind.
    """  
    value = np.sum(ind)  
    return value

def trap5( ind, k = 5): 
    """
        Hàm đánh giá trap: cài đặt bẫy khi đếm số bit 1 trong chuỗi nhị phân của từng cá thể

        Arguments:
        k -- số lượng cá thể trong 1 block
        sub_inds -- số lượng block (bẫy) có trong quần thể
        ind -- cá thể cần được đánh giá

        Returns:
        value -- giá trị của quần thể đó 
    """  
    sub_individuals = np.reshape(ind, (len(ind) // k, k))
    result = 0
    for individual in sub_individuals: 
        num_ones = sum(individual)  
        f_trap = (k * (num_ones == k)) + (((k - 1) - num_ones) * (num_ones < k))  
    
        result += f_trap
    return result

def evaluate_population( pop, option ): 
    """
    Hàm đánh giá tất cả cá thể trong quần thể.
    
    Arguments:
    pop -- Quần thể cần được đánh giá.

    Returns:
    values -- Giá trị của tất cả các cá thể trong quần thể.
    """  
    global accumulative_call_fitness_evaluate
    accumulative_call_fitness_evaluate += 1

    if option == 'onemax':
        values = [onemax(ind) for ind in pop]
    if option == 'trap5': 
        values = [trap5(ind) for ind in pop] 
    
    return values

def better_fitness( fitness_1, fitness_2, maximization=True ): 
    """
    Hàm so sánh độ thích nghi của 2 cá thể.
    
    Arguments:
    fitness_1 -- Độ thích nghi của cá thể 1.
    fitness_2 -- Độ thích nghi của cá thể 2.
    maximization -- Biến boolean cho biết bài toán đang giải thuộc dạng tối đa hoá (mặc định) hay không
    
    Returns:
    True nếu fitness_1 tốt hơn fitness_2. Ngược lại, trả về False.
    """ 
    if maximization:
        if fitness_1 > fitness_2:
            return True
    else:
        if fitness_1 < fitness_2:
            return True
        
    return False

def tournament_selection( pop, pop_fitness, selection_size, tournament_size, option): 
    """
    Hàm chọn lọc cạnh tranh.
    
    Arguments:
    pop -- Quần thể để thực hiện phép chọn lọc.
    pop_fitness -- Mảng 1 chiều chứa giá trị (độ thích nghi) của từng cá thể trong quần thể. 
    selection_size -- Số lượng cá thể sẽ được chọn.
    tournament_size -- Kích thước của tournament: Số lượng các cá thể được so sánh với nhau mỗi lần.
    
    Returns:
    selected_indices -- Chỉ số của những cá thể trong quần thể pop được chọn. Chỉ số có thể được lặp lại.
    """
    num_individuals = len(pop)
    indices = np.arange(num_individuals)
    selected_indices = []
    
    while len(selected_indices) < selection_size: 
        np.random.shuffle(indices)  
        for i in range(0, num_individuals // tournament_size):  
            indice_board = indices[i*tournament_size:(i+1)*tournament_size] 
            board = [pop[ind] for ind in indice_board]
            best_fit = np.argmax(evaluate_population(board, option))
            selected_indices.append(indice_board[best_fit]) 
       
    return selected_indices[: selection_size]

def variation( pop , variate): 
    """
    Hàm biến đổi tạo ra các cá thể con.
    
    Arguments:
    pop -- Quần thể hiện tại.

    Returns:
    offspring -- Quần thể chứa các cá thể con được sinh ra.
    """  
    num_individuals = len(pop)
    num_parameters = len(pop[0])
    indices = np.arange(num_individuals) 
    np.random.shuffle(indices)
    offspring = []
    
    for i in range(0, num_individuals, 2):
        idx1 = indices[i]
        idx2 = indices[i+1]
        offspring1 = list(pop[idx1])
        offspring2 = list(pop[idx2])
        
        if variate == 'ux':
            for idx in range(0, len(offspring1)): 
                prob = np.random.rand() 
                if prob > 0.5: 
                    offspring1[idx], offspring2[idx] = offspring2[idx], offspring1[idx] 
        
        if variate == '1x':
            index = np.random.randint(len(offspring1))
            for idx in range(index, len(offspring1)):
                offspring1[idx], offspring2[idx] = offspring2[idx], offspring1[idx] 

        offspring.append(offspring1)
        offspring.append(offspring2)

    offspring = np.array(offspring)
    return offspring

def isconvergence(pop_fitness): 
    """ 
    Hàm kiểm tra sự hội tụ của quần thể
    """
    return np.all(pop_fitness == pop_fitness[0])

def popop(num_individuals, num_parameters, option, variate): 
    """
    Hàm cài đặt thuật giải di truyền theo các bước P->O->(P+O)->P
    
    Arguments:
        num_individuals -- Số lượng cá thể trong quần thể.
        num_parameters -- Số lượng biến.
        num_generations -- Số thế hệ thuật toán sẽ chạy.

    Returns:
        Xác nhận quần thể đã hội tụ tại cá thể tốt nhất (ind = 1)
    """  
    pop = initialize_population(num_individuals, num_parameters)
    pop_fitness = evaluate_population(pop, option) 
     
    selection_size = len(pop)
    tournament_size = 4

    while isconvergence(pop_fitness) == False: 
        offspring = variation(pop, variate)
        offspring_fitness = evaluate_population(offspring, option)
 
        pool = np.vstack((pop,offspring))
        pool_fitness = np.hstack((pop_fitness, offspring_fitness))

        pool_indices = tournament_selection(pool, pool_fitness, selection_size, tournament_size, option)

        pop = pool[pool_indices, :]
        pop_fitness = pool_fitness[pool_indices] 

    return np.sum(pop_fitness) == num_parameters * num_individuals

def progress1(num_parameters, set_random_seed, option, variate): 
    n_upper = 4
    logger = {} 

    while True:
        n_upper <<= 1   
        accumulative_true = 0 
        for random_seed in set_random_seed: 
            np.random.seed(random_seed)  
            checker = popop(num_individuals=n_upper, num_parameters=num_parameters, option=option, variate=variate)
            accumulative_true += checker == True
        
        if accumulative_true == len(set_random_seed): 
            break  
    return n_upper

def progress2(num_parameters, set_random_seed, n_upper, option, variate): 
    n_lower = n_upper // 2 
    while (n_upper - n_lower) / n_upper > 0.1: 
        n = (n_upper + n_lower) // 2

        accumulative_true = 0 
        global accumulative_call_fitness_evaluate
        accumulative_call_fitness_evaluate = 0

        for random_seed in set_random_seed: 
            np.random.seed(random_seed)  
            checker = popop(num_individuals=n, num_parameters=num_parameters, option=option, variate=variate)
            accumulative_true += checker == True

        success = accumulative_true == len(set_random_seed)
        
        if success == True: 
            n_upper = n 
        else: 
            n_lower = n  
        average_number_of_evaluation = accumulative_call_fitness_evaluate / 10
        if (n_upper - n_lower) <= 2:  
            break

    return n_upper, average_number_of_evaluation

def run_all_bisection(l, mssv, eva, var): 
    set_random_seed = initialize_random_seed(mssv)
    result = [] 
    count = 0
    for collection_random_seed in set_random_seed:
        count += 1
        print('Running {}-th set seed'.format(count))
        start = time.time()
        n_upper = progress1(l, collection_random_seed, eva, var)
        n, avg = progress2(l, collection_random_seed, n_upper, eva, var) 
        end = time.time()
        print('Complete in {} seconds'.format(end - start))
        print('--------------------------')
        result.append([n, avg])
    return result   
    
l = [40, 80, 160]
opt = 'trap5'
var = 'ux'

for length in l:
    print('-------------------{}------------------'.format(length))
    accumulative_call_fitness_evaluate = 0 
    review = run_all_bisection(length, 19520448, opt, var)
    
    np.savetxt('data_{}_{}_{}_twel.csv'.format(length, opt, var), review, delimiter=',', fmt='%.2f')
    
