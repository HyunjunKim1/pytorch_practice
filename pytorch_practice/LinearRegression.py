#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#
#torch.manual_seed(1)
#
#x_train = torch.FloatTensor([[1], [2], [3]])
#y_train = torch.FloatTensor([[2], [4], [6]])
#
#W = torch.zeros(1, requires_grad=True)
#b = torch.zeros(1, requires_grad=True)
#
#optimizer = optim.SGD([W, b], lr=0.01)
#
#nb_epochs = 1999 
#for epoch in range(nb_epochs + 1):
#
#    hypothesis = x_train * W + b
#
#    cost = torch.mean((hypothesis - y_train) ** 2)
#
#    optimizer.zero_grad()
#    cost.backward()
#    optimizer.step()
#
#    if epoch % 100 == 0:
#        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
#            epoch, nb_epochs, W.item(), b.item(), cost.item()
#        ))

# nn.Module로 구현

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# input_dim = 1, output_dim = 1
model = nn.Linear(1,1)

#print(list(model.parameters()))

# 경사하강법
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)
    optimizer.zero_grad()

    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)

print(pred_y)