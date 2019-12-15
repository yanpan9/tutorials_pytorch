import torch

from utils import loss_curve

def generate_data(x, w, b):
    y = w*x+b+0.01*torch.randn_like(x)
    return y

    
def predict(x, w, b):
    return w*x+b

if __name__ == "__main__":
    x = torch.arange(0, 15, 0.1)
    y = generate_data(x, 1, 0)

    w = torch.randn((1), requires_grad=True)
    b = torch.randn((1), requires_grad=True)

    mse = torch.nn.MSELoss()
    lr = 1e-3

    losses = list()

    for i in range(100):   
        pred = predict(x, w, b)
        loss = mse(y, pred)
        if i%10==0:
            print("At step %i, the loss is %.4f"%(i, loss))
        losses.append(loss.item())
        if w.grad:
            w.grad.data.zero_()
            b.grad.data.zero_()
        loss.backward()
        # Update data of the variable
        # not variable
        w.data -= lr*w.grad
        b.data -= lr*b.grad
    
    loss_curve(losses)