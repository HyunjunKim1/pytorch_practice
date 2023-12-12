import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

## 점수 훈련 데이터
#x_train  =  torch.FloatTensor([[73,  80,  75], 
#                               [93,  88,  93], 
#                               [89,  91,  80], 
#                               [96,  98,  100],   
#                               [73,  66,  70]])  
#y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
#
#W = torch.zeros((3, 1), requires_grad=True)
#b = torch.zeros(1, requires_grad=True)
#
#optimizer = optim.SGD([W, b], lr=1e-5)
#
#
#nb_epochs = 200
#for epoch in range(nb_epochs + 1):
#
#    # H(x) 계산
#    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
#    hypothesis = x_train.matmul(W) + b
#
#    # cost 계산
#    cost = torch.mean((hypothesis - y_train) ** 2)
#
#    # cost로 H(x) 개선
#    optimizer.zero_grad()
#    cost.backward()
#    optimizer.step()
#
#    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
#        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
#    ))

# 다중선형회귀 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

new_var =  torch.FloatTensor([[73, 80, 75]]) 
pred_y = model(new_var) 
print(pred_y)