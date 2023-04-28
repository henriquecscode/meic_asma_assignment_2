import gymnasium as gym

# Create the environment
env = gym.make('Ant-v4', render_mode="human" )  # continuous: LunarLanderContinuous-v2
# env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2

# required before you can step the environment
env.reset()

for j in range(100):
    # sample action:
	env.step(env.action_space.sample())
	env.render()