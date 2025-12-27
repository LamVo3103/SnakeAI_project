import matplotlib.pyplot as plt
from IPython import display
from agent_dqn import Agent
from game import SnakeGameAI

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(.1)

def train():
    plot_scores, plot_mean_scores, total_score, record = [], [], 0, 0
    agent, game = Agent(), SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            game.record = record
            
            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')
            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__': train()