import torch.nn
from torch import Tensor
import numpy as np
from model import flood_comp_model
from torch.optim import Adam

class train():

    def __init__(self):
        self.dataset = np.load('new_dataset.npy', allow_pickle=True)
        self.event_dictionary = np.load('event_dictionary.npy', allow_pickle=True)
        self.model = flood_comp_model()
        self.optimizer = Adam(params=self.model.parameters(), lr=0.01)
        self.loss = torch.nn.MSELoss()


    def training_loop(self, epochs):
        self.model.eval()
        for epoch in range(0, epochs):
            np.random.shuffle(self.dataset)
            epoch_loss = 0
            for i, observation in enumerate(self.dataset):
                try:
                    # establish observation and GT
                    print("on input " + str(i))
                    input = observation[0: len(observation) - 1]
                    input = torch.tensor(input.astype(np.float32), dtype=torch.float32)
                    ground_truth = observation[-1]
                    ground_truth = torch.tensor(ground_truth.astype(np.float32),  dtype=torch.float32)
                    print(input)
                    print(ground_truth)

                    # zero gradients
                    self.optimizer.zero_grad()

                    # get model prediction
                    prediction = self.model(input)

                    # calculate loss
                    loss = self.loss(prediction, ground_truth)
                    epoch_loss += loss.item()

                    # backpropagate
                    loss.backward()

                    # step
                    self.optimizer.step()
                except:
                    print("observation "+ str(i) + " caused an exception")
            print(epoch_loss)
        torch.save(self.model.state_dict(), 'model_weights.pt')
        

if __name__ == '__main__':
    # obj = train()
    # obj.training_loop(5)
    print(np.load('new_dataset.npy', allow_pickle=True)[-2])

        
