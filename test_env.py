#Manual implementation/control for LunarLander -v2
from utils import *
from training_utils import *
from modified_env_utils import *
import pygame

def action_key_press():
    keys = pygame.key.get_pressed()
    action = np.array([0, 0])
    if keys[pygame.K_UP]:
        action[0]= 1
    if keys[pygame.K_LEFT]:
        action[1]-=1
    if keys[pygame.K_RIGHT]:
        action[1]+=1
    return action, keys
    
params = {}
# Using our environment wrappers
env = get_wrapped_lunar_environment(**params)
end = False
while not end:
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, keys = action_key_press()
        if(keys[pygame.K_ESCAPE]):
            end = True
            break
        obs, reward, done, info = env.step(action)

print("Done. Exiting...")