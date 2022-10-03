import numpy as np
import random
from tensorflow import keras
import matplotlib.pyplot as plt
import pymunk

CROSSOVER_RATE = 0.2
MUTATION_RATE = 0.05

def normalise_and_prep(state):
    norm_pos = state[0] / 800 
    norm_velocity = state[1] / 1500
    arr = np.asarray([norm_pos, norm_velocity])
    return arr.reshape(1, 2)

def run_test_with_model(model):

  space = pymunk.Space()
  space.gravity = 0, 2001 

  b0 = space.static_body

  segment_1 = pymunk.Segment(b0, (0, 800), (640, 800), 4)
  segment_1.elasticity = 0.5

  segment_2 = pymunk.Segment(b0, (0, 0), (640, 0), 4)
  segment_2.elasticity = 0.5

  body = pymunk.Body(mass=4, moment=10)
  body.position = 100, 40

  circle = pymunk.Circle(body, radius=20)
  circle.elasticity = 0.95

  space.add(body, circle, segment_1, segment_2)

  #print("starting")

  reward = 0
  running = True
  frames_passed = 0
  
  while running:

      space.step(0.01)
      frames_passed += 1

      curr_position = circle.body.position[1]
      curr_velocity = circle.body.velocity[1]

      if frames_passed % 60 == 0: #50 frames = half a second

          state = normalise_and_prep([curr_position, curr_velocity])
          force_size = model.predict(state)
          #print(force_size)
          force = -800000 * force_size
          circle.body.apply_force_at_local_point((0, force), (0, 0)) #this code can add a force to the body:

      if 200 < curr_position < 600: #this is how we decide a reward:
          reward += 1
        
      if 350 < curr_position < 450: #this is how we decide a reward:
          reward += 1
      

      if curr_position < 20 or  curr_position > 775 or frames_passed > 10000:  #if it hits the floor or goes too high
          running = False
  #print(reward)
  return reward




def create_random_model(input_shape, num_outputs):

  curr_model = keras.models.Sequential([
                                  keras.layers.Dense(6, input_shape=[input_shape]),   
                                  keras.layers.Dense(10, activation='relu'),
                                  keras.layers.Dense(10, activation='relu'),
                                  keras.layers.Dense(num_outputs, activation='sigmoid') #changed from 2
            ])
  
  model_weights = []
  for layer_index in range(len(curr_model.layers)):
    model_data = curr_model.layers[layer_index].get_weights()[0]

    for i in range(len(model_data)):
      for j in range(len(model_data[i])):
        model_data[i][j] = random.uniform(-1, 1)

    weight_framework = curr_model.layers[layer_index].get_weights()
    weight_framework[0] = model_data
    curr_model.layers[layer_index].set_weights(weight_framework)
    #print(weight_framework)

  return curr_model


def initialize_population(population_size):
  initial_generation = []

  for i in range(population_size):
    curr_model = create_random_model(2, 1)
    
    fittness_of_model = run_test_with_model(curr_model)
    initial_generation.append({
        'model':curr_model,
        'fitness':fittness_of_model
    })

    #print(initial_generation[i])

    

  return initial_generation 

def perform_tournement_of_4(population):
  tournement_members = []
  temp_population = population.copy()


  for i in range(4):
    tournement_members.append(temp_population.pop(random.randrange(len(temp_population))))

  if tournement_members[0]['fitness'] < tournement_members[1]['fitness']:
    tournement_members.pop(0)
  else:
    tournement_members.pop(1)

  if tournement_members[1]['fitness'] < tournement_members[2]['fitness']:
    tournement_members.pop(1)
  else:
    tournement_members.pop(2)

  return tournement_members

def perform_tornement(number_of_participants, population):
  final_pool = []
  number_iterations_of_4 = number_of_participants / 4
  for i in range(int(number_iterations_of_4)):
    pool_member_1, pool_member_2 = perform_tournement_of_4(population)
    final_pool.append(pool_member_1)
    final_pool.append(pool_member_2)
  
  return perform_tournement_of_4(final_pool)

def create_children(parents, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE):

  parent_1 = parents[0]['model']
  parent_2 = parents[1]['model']

  child_framework_1 = create_random_model(2, 1)
  child_framework_2 = create_random_model(2, 1)

  for layer_index in range(len(parent_1.layers)):
    
    child_1_weights = parent_1.layers[layer_index].get_weights()[0]
    child_2_weights = parent_2.layers[layer_index].get_weights()[0]
    
    child_1_data = parent_1.layers[layer_index].get_weights()[0]
    child_2_data = parent_2.layers[layer_index].get_weights()[0]

    for i in range(len(child_1_data)):
      for j in range(len(child_1_data[i])):

        if random.uniform(0, 1) > 0.5:
          child_1_weights[i][j] = child_1_data[i][j]
        else:
          child_1_weights[i][j] = child_2_data[i][j]

        if random.uniform(0, 1) > 0.5:
          child_2_weights[i][j] = child_1_data[i][j]
        else:
          child_2_weights[i][j] = child_2_data[i][j]

        #mutation
        if random.uniform(0, 1) < mutation_rate:
          child_1_weights[i][j] =  - child_1_weights[i][j]
        
        if random.uniform(0, 1) < mutation_rate:
          child_2_weights[i][j] =  - child_2_weights[i][j]


    new_weight_1 = child_framework_1.layers[layer_index].get_weights()
    new_weight_1[0] = child_1_weights

    new_weight_2 = child_framework_2.layers[layer_index].get_weights()
    new_weight_2[0] = child_2_weights

    child_framework_1.layers[layer_index].set_weights(new_weight_1)
    child_framework_2.layers[layer_index].set_weights(new_weight_2)

  return child_framework_1, child_framework_2

def display_and_return_metrics(population, generation, number_of_elites, number_of_random, best_fitness, time_since_last_best_fitness):
  avg_fitness = 0
  avg_fitness_excluding_elites = 0
  stop_early = False

  for member in population:
    avg_fitness += member['fitness']
  avg_fitness = avg_fitness / len(population)
  

  for member in population[:-(number_of_elites + number_of_random)]:
    avg_fitness_excluding_elites += member['fitness']
  avg_fitness_excluding_elites = avg_fitness_excluding_elites / (len(population) - number_of_elites - number_of_random) 

  print(f'GENERATION: {generation}, AVG FITNESS: {avg_fitness}, POPULATION SIZE: {len(population)}, FITNESS EXCLUDING ELITES: {avg_fitness_excluding_elites}')
  print(population)
  print('====================================================================================================')

  #printing the best model weights:
  if generation % 10 == 0 and generation is not 0:
    print(population[-(number_of_elites + number_of_random)])
    #for layer in population[-10]['model'].layers:
      #print(layer.get_weights())
    print('====================================================================================================')
  
  #early_stopping architecture
  if avg_fitness > best_fitness:
    best_fitness = avg_fitness
    time_since_last_best_fitness = 0

  time_since_last_best_fitness += 1

  if time_since_last_best_fitness > 14:
    stop_early = True

  return avg_fitness, avg_fitness_excluding_elites, stop_early, best_fitness, time_since_last_best_fitness

def create_new_generation(previous_gen, number_of_prev_elites, number_of_random, population_size):
  generation = []
  number_of_reps = int((population_size - number_of_prev_elites - number_of_random) /2)

  for i in range(number_of_reps): 
    tornement_winners = perform_tornement(8, previous_gen)
    child_1, child_2 = create_children(tornement_winners)

    fittness_of_child_1 = run_test_with_model(child_1)  
    fittness_of_child_2 = run_test_with_model(child_2)  

    generation.append({
        'model':child_1,
        'fitness':fittness_of_child_1
    })

    generation.append({
        'model':child_2,
        'fitness':fittness_of_child_2
    })

  #last we need to get the best n from the prev generation 
  previous_gen_sorted = sorted(previous_gen, key=lambda d: d['fitness'])
  for i in range(number_of_prev_elites):
    elite = previous_gen_sorted.pop()
    generation.append(elite)
  
  for i in range(number_of_random):
    curr_model = create_random_model(2, 1)
    
    fittness_of_model = run_test_with_model(curr_model)
    generation.append({
        'model':curr_model,
        'fitness':fittness_of_model
    })

  return generation


NUMBER_OF_GENERATIONS = 15
SIZE_OF_POPULATION = 40
NUMBER_OF_ELITES = 8
NUMBER_OF_RANDOM = 0
fitness_history = []
fitness_history_excluding_elites = []
best_fitness = 0
time_since_last_best_fitness = 0

population = initialize_population(SIZE_OF_POPULATION)  #I have added on the v2 for messing around purposes 
curr_avg_fitness, curr_avg_fitness_excluding_elites, stop_early, best_fitness, time_since_last_best_fitness  = display_and_return_metrics(population, 0, NUMBER_OF_ELITES, NUMBER_OF_RANDOM, best_fitness, time_since_last_best_fitness)
fitness_history.append(curr_avg_fitness)
fitness_history_excluding_elites.append(curr_avg_fitness_excluding_elites)

for generation in range(1, NUMBER_OF_GENERATIONS):
  population = create_new_generation(population, NUMBER_OF_ELITES, NUMBER_OF_RANDOM, SIZE_OF_POPULATION)
  curr_avg_fitness, curr_avg_fitness_excluding_elites, stop_early, best_fitness, time_since_last_best_fitness = display_and_return_metrics(population, generation, NUMBER_OF_ELITES, NUMBER_OF_RANDOM, best_fitness, time_since_last_best_fitness)

  fitness_history.append(curr_avg_fitness)
  fitness_history_excluding_elites.append(curr_avg_fitness_excluding_elites)
  if stop_early:
    break
  