import torch
from agent_dqn import Agent
from game import SnakeGameAI
import time

def test():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    agent = Agent()
    try:
        agent.model.load_state_dict(torch.load('model/model.pth', map_location=device, weights_only=True))
        agent.model.eval()
        print("Model loaded!")
    except:
        print("Model not found!"); return

    current_record = 0
    game = SnakeGameAI()
    while True:
        reward, done, score = game.play_step(agent.get_action(agent.get_state(game), is_test=True))
        if score > current_record:
            current_record = score
            game.record = current_record

        if done:
            print(f'Test Score: {score}')
            game.reset()
            time.sleep(1)

if __name__ == '__main__': test()