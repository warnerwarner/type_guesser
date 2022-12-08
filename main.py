#%%
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import trange

#%%
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
#%%
model = nn.Sequential(nn.Linear(n_in, n_h),
                        nn.ReLU(),
                        nn.Linear(n_h, n_out),
                        nn.Sigmoid())


# %%
criterion = torch.nn.MSELoss()
# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# %%
for epoch in range(50):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(x)

   # Compute and print loss
   loss = criterion(y_pred, y)
   print('epoch: ', epoch,' loss: ', loss.item())

   # Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()

   # perform a backward pass (backpropagation)
   loss.backward()

   # Update the parameters
   optimizer.step()
# %%

import requests

# %%

def get_pokemon(main_address="https://www.serebii.net/pokemon/art",
                output_folder='pokemon_images',
                end=1008):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    for i in trange(end+1, position=0, leave=False):
        
        request = requests.get(main_address+f'/{i:03d}.png')
        if request.status_code == 200:
            img=request.content
            with open(f'pokemon_images/{i:03d}.png', 'wb') as writer:
                writer.write(img)


# %%
get_pokemon()
# %%
from bs4 import BeautifulSoup
# %%
import matplotlib.pyplot as plt
# %%
plt.plot(np.arange(10))
# %%
p_info =     