import learn2learn as l2l
import torch
import torch.nn as nn
import torch.optim as optim
import metaworld
import random
import numpy as np

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Meta-World
    mt1 = metaworld.MT1('pick-place-v2')  # Ensure the task name is correct
    env = mt1.train_classes['pick-place-v2']()  # Instantiate the environment
    task = random.choice(mt1.train_tasks)
    env.set_task(task)

    # Setup the model
    model = nn.Sequential(
        nn.Linear(np.prod(env.observation_space.shape), 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, env.action_space.shape[0]),
    )
    model.to(device)

    # MAML setup
    maml = l2l.algorithms.MAML(model, lr=0.01).to(device)
    optimizer = optim.Adam(maml.parameters(), lr=0.001)

    # Training loop
    for iteration in range(100):
        optimizer.zero_grad()
        meta_loss = 0  # This will accumulate scalar loss values

        # Clone the model for task-specific adaptation
        learner = maml.clone()

        # Adaptation phase
        for step in range(5):
            observation, _ = env.reset()
            observation_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
            action = learner(observation_tensor)

            _, reward, terminated, truncated, info = env.step(action.squeeze().detach().cpu().numpy())

            # Use a dummy loss for adaptation that's differentiable and involves the model
            dummy_loss = action.pow(2).mean()
            learner.adapt(dummy_loss)

        # Meta-update
        for step in range(5):
            observation, _ = env.reset()
            observation_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
            action = learner(observation_tensor)
            _, reward, terminated, truncated, info = env.step(action.squeeze().detach().cpu().numpy())

            # Now, ensure the meta_loss is a tensor that can accumulate gradients
            if step == 0:  # For the first step, initialize meta_loss as a tensor
                meta_loss = action.pow(2).mean()
            else:  # For subsequent steps, add the loss to meta_loss
                meta_loss += action.pow(2).mean()

        # Backward and optimize
        meta_loss.backward()
        optimizer.step()

        print(f"Iteration {iteration}: Meta loss {meta_loss.item()}")

if __name__ == '__main__':
    main()
