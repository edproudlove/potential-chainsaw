import numpy as np
import random
import pymunk
import pygame
import pymunk.pygame_util
import requests 

pygame.init()
size = 640, 800
screen = pygame.display.set_mode(size)

draw_options = pymunk.pygame_util.DrawOptions(screen)
        

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

reward = 0
running = True
frames_passed = 0


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #this is setup for the pygame look
    screen.fill((125, 125, 125))

    pygame.draw.line(screen, (0, 0, 0), (0, 200), (640, 200))
    pygame.draw.line(screen, (0, 0, 0), (0, 600), (640, 600))

    space.debug_draw(draw_options)
    pygame.display.update()
    space.step(0.01)
    frames_passed += 1

    curr_position = circle.body.position[1]
    curr_velocity = circle.body.velocity[1]

    if frames_passed % 50 == 0: #50 frames = half a second

        r = requests.post('http://127.0.0.1:5000/', json = {'position': curr_position / 800, 'velocity': curr_velocity / 1500}) #normalised woth divides
        force_size = float(r.json()['algo_output'])

        print(force_size)
        
        force = -800000 * force_size
        circle.body.apply_force_at_local_point((0, force), (0, 0)) #this code can add a force to the body:


    if 200 < curr_position < 600: #this is how we decide a reward:
        reward += 1

    if 350 < curr_position < 450: #this is how we decide a reward:
          reward += 1


    if curr_position < 25 or  curr_position > 775:  #if it hits the floor or goes too high
        running = False
    

pygame.quit()
print(reward)
print(frames_passed)
