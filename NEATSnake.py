from __future__ import print_function
from math import sqrt
from random import randint
from neat import nn, population, statistics
import os
import curses

def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)
        score, turns = play_game(net)
        #print ("Score, turns", score, turns)
        g.fitness = score + turns*10

def play_game(net):
    """
    Simulates game with net's outputs for controls
    
    Args:
        net: Neural Net that inputs can be thrown to to get an output
    Returns:
        int: Score earned
    """
    turns = 0
    score = 0
    snake = [[4,10], [4,9], [4,8]]
    food = [10,20]
    moves_since_food = 0
    while moves_since_food < 75:
        turns += 1
        moves_since_food += 1
        food_dist = dist_to_food(snake, food)

        inputs = [food_dist, dist_to_body(snake, 1, 0), dist_to_body(snake, -1, 0), 
                  dist_to_body(snake, 0, 1), dist_to_body(snake, 0, -1)]
        direction = int(net.serial_activate(inputs)[0] * 4.0) 
        snake.insert(0, [snake[0][0] + (direction == 0 and 1) + (direction == 1 and -1), snake[0][1] \
                        + (direction == 2 and -1) + (direction >= 3 and 1)])

        score -= food_dist

        #print("Direction, snake, food", direction, snake[0], food)

        if snake[0][0] == 0: snake[0][0] = 18
        if snake[0][1] == 0: snake[0][1] = 58
        if snake[0][0] == 19: snake[0][0] = 1
        if snake[0][1] == 59: snake[0][1] = 1
        if snake[0] in snake[1:]: break
        
        if snake[0] == food:
            moves_since_food = 0
            food = []
            score += 100000
            while food == []:
                food = [randint(1, 18), randint(1, 58)]
                if food in snake: food = []
            #win.addch(food[0], food[1], '*')
        else:    
            last = snake.pop()                                          # [1] If it does not eat the food, length decreases
            #win.addch(last[0], last[1], ' ')
        #win.addch(snake[0][0], snake[0][1], '#')

    return score, turns
        
def dist_to_body(snake, dx, dy):
    pos = snake[0]
    pos[0] += dx
    pos[1] += dy

    dist = 0
    while pos not in snake:
        dist += 1
        pos[0] += dx
        pos[1] += dy
    return dist

def dist_to_food(snake, food):
    sx, sy = snake[0]
    fx,fy = food
    
    return sqrt((fx-sx)**2 + (fy-sy)**2) 


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'snake_config')
pop = population.Population(config_path)
pop.run(eval_fitness, 3000)

# Log statistics.
statistics.save_stats(pop.statistics)
statistics.save_species_count(pop.statistics)
statistics.save_species_fitness(pop.statistics)

print('Number of evaluations: {0}'.format(pop.total_evaluations))

# Show output of the most fit genome against training data.
winner = pop.statistics.best_genome()
winner_net = nn.create_feed_forward_phenotype(winner)
print('\nBest genome:\n{!s}'.format(winner))

curses.initscr()
win = curses.newwin(20, 60, 0, 0)
win.keypad(1)
curses.noecho()
curses.curs_set(0)
win.border(0)
win.nodelay(1)

score = 0

snake = [[4,10], [4,9], [4,8]]                                     # Initial snake co-ordinates
food = [10,20]                                                     # First food co-ordinates

win.addch(food[0], food[1], '*')                                   # Prints the food

while True:                                                   # While Esc key is not pressed
    win.border(0)
    win.addstr(0, 2, 'Score : ' + str(score) + ' ')                # Printing 'Score' and
    win.addstr(0, 27, ' SNAKE ')                                   # 'SNAKE' strings
    win.timeout(150 - (len(snake)/5 + len(snake)/10)%120)          # Increases the speed of Snake as its length increases
    
    food_dist = dist_to_food(snake, food)
    inputs = [food_dist, dist_to_body(snake, 1, 0), dist_to_body(snake, -1, 0), 
              dist_to_body(snake, 0, 1), dist_to_body(snake, 0, -1)]
    direction = int(winner_net.serial_activate(inputs)[0] * 4.0) 
    
    snake.insert(0, [snake[0][0] + (direction == 0 and 1) + (direction == 1 and -1), snake[0][1] \
                    + (direction == 2 and -1) + (direction >= 3 and 1)])

    # If snake crosses the boundaries, make it enter from the other side
    if snake[0][0] == 0: snake[0][0] = 18
    if snake[0][1] == 0: snake[0][1] = 58
    if snake[0][0] == 19: snake[0][0] = 1
    if snake[0][1] == 59: snake[0][1] = 1

    # Exit if snake crosses the boundaries (Uncomment to enable)
    #if snake[0][0] == 0 or snake[0][0] == 19 or snake[0][1] == 0 or snake[0][1] == 59: break

    # If snake runs over itself
    if snake[0] in snake[1:]: break

    
    if snake[0] == food:                                            # When snake eats the food
        food = []
        score += 1
        while food == []:
            food = [randint(1, 18), randint(1, 58)]                 # Calculating next food's coordinates
            if food in snake: food = []
        win.addch(food[0], food[1], '*')
    else:    
        last = snake.pop()                                          # [1] If it does not eat the food, length decreases
        win.addch(last[0], last[1], ' ')
    win.addch(snake[0][0], snake[0][1], '#')
    
curses.endwin()
print("\nScore - " + str(score))