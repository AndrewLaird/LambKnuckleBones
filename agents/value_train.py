from agents import Agent, RandomAgent, ModelAgent, ValueAgent
from arena import run_arena

if __name__ == "__main__":
    random_agent = RandomAgent()
    value_agent = ValueAgent()

    value_agent.load("value_agent.pt")

    all_training_data = []
    epochs = 0
    while True:  # goodbye cpu
        print(epochs)
        # play all 100 games at once
        value_agent.value_model.eval()
        for i in range(500):
            winner, training_data = run_arena(value_agent, value_agent, render=False)
            all_training_data.extend(training_data)
        value_agent.value_model.train(mode=True)
        value_agent.train_with_data(all_training_data)
        value_agent.save("value_agent.pt")
        epochs += 1
        all_training_data = []
