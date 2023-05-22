from utils import *
from training_utils import *
from modified_env_utils import *
import pygame

def action_key_press():
    keys = pygame.key.get_pressed()
    action = np.array([0, 0])
    if keys[pygame.K_UP]:
        action[0]= 1
    elif keys[pygame.K_LEFT]:
        action[1]-=1
    elif keys[pygame.K_RIGHT]:
        action[1]+=1
    return action, keys
    
params = {}
env = get_wrapped_lunar_environment(**params)
while True:
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, keys = action_key_press()
        if(keys[pygame.K_ESCAPE]):
            exit(0)
        obs, reward, done, info = env.step(action)
        # if reward > 0:
        #     print(f"Reward: {reward}")    

print("Done. Exiting...")