import torch




def alpine2D(x1, x2=torch.tensor(4.), noise=0.3):
    x1 = x1.reshape(-1,1)
    if noise==0.:
        return (torch.sin(x1) * torch.sin(x2) * torch.sqrt(x1*x2)).reshape(-1)
    else:
        if x1.shape != x2.shape:
            x2 = x2.repeat(x1.shape[0],1)
        res = []
        for x1i, x2i in zip(x1,x2):
            res.append(torch.sin(x1i) * torch.sin(x2i) * torch.sqrt(x1i * x2i) + torch.distributions.Normal(0,noise).sample())
            
        return torch.cat(res).reshape(-1)
#        return (torch.cat([torch.sin(x1i) * 
#                           torch.sin(x2i) * 
#                           torch.sqrt(x1i*x2i) + 
#                           torch.distributions.Normal(0,noise).sample() for x1i, x2i in zip(x1.flatten(),
#                                                                                            x2.flatten())])).reshape(-1)
    
def alpine2D_df(x1, x2=torch.tensor(4.), noise=0.3):
    x1 = x1.round()
    
    return alpine2D(x1=x1, x2=x2, noise=noise)


def alpine2D_ds(x1, x2, noise=0.3):
    x2 = x2.round()
    
    return alpine2D(x1=x1, x2=x2, noise=noise)

    
def Objective2D(x1, x2=0., noise=0.3):
    def inner_function(x1, x2):
        res = 3. * (1.-x1)**2 * torch.exp(-(x1**2) - (x2 + 1.)**2)
        res = res - 10. * (x1/5. - x1**3 - x2**5) * torch.exp(-x1**2 - x2**2)
        res = res - 1/3. * torch.exp(-(x1 + 1.)**2 - x2**2) 
        return res.reshape(-1)
    
    x1 = x1 - 2.5
    if noise==0.:
        return inner_function(x1, x2).reshape(-1)
    else:
        if x2.shape == torch.Size([]):
            x2 = x2.repeat(x1.shape[0])
            
        res = [inner_function(x1i, x2i) + torch.distributions.Normal(0,noise).sample() for x1i, x2i in zip(x1, x2)]
        return torch.cat((res)).reshape(-1)
