import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(ANN, self).__init__()
        #useful in 2 step inheritance
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out_by_input = self.fc1(x)
        
        out_by_relu = self.relu(out_by_input)
        
        out_by_hidden = self.fc2(out_by_relu)
        
        return out_by_hidden
    
    
if __name__=="__main__":
    input_size = 784
    hidden_size = 256
    output_size = 10
    
    model = ANN(input_size, hidden_size, output_size)
    
    criterion = nn.CrossEntropyLoss()
    
    lr = 0.01
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    inputs =torch.randn(100,input_size)
    labels =torch.randint(0,output_size, (100,))
    
    n_epochs=10
    
    for epoch in range(n_epochs):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        opt.zero_grad()
        
        loss.backward()
        opt.step()
        
        if epoch % 10 == 9:
            print(loss)