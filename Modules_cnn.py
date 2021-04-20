from Series2GAF import gaf_encode
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=2)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        return output


def state_convert(ohlc):
    
    O, H, L, C = gaf_encode(ohlc['Open']), gaf_encode(ohlc['High']), \
                    gaf_encode(ohlc['Low']),gaf_encode(ohlc['Close'])
    
    state = np.stack((O,H,L,C),axis=-1)
    state = torch.from_numpy(state).unsqueeze(0).permute(0, 3, 1, 2)
    # 要 as contiguous 才能用 .view()
    state = np.ascontiguousarray(state, dtype=np.float32)
    state = torch.from_numpy(state).to(device)
    return state

def market_step(data,action,timestep,order,window_size,max_bid,last_ungain,sortino):
    new_close=data.iloc[timestep-1]['Close']
    order_flag=False
    order_count=0
    long_short=0
    ungain=0
    commision=0
    #max_bid=1000
    

    for ord in order:
        if not ord['close']:
            ungain-=commision
            if ord['type']=='buy':
                long_short+=1
            elif ord['type']=='sell':
                long_short-=1
            order_count+=1
        elif ord['close']:
            ungain-=commision*2

    if action==2:
        
        for ord in order:
            if ord['type']=='sell' and not ord['close']:
                order_flag=True
                ungain-=commision
                ord['close']=True
                ord['gain']=ord['price']-new_close
                break
        if not order_flag and order_count<max_bid:
            ungain-=commision
            new_order={
                'id':timestep,
                'type':'buy',
                'price':new_close,
                'gain':0,
                'close':False
            }
    elif action==0:
        for ord in order:
            if ord['type']=='buy' and not ord['close']:
                order_flag=True
                ungain-=commision
                ord['close']=True
                ord['gain']=new_close-ord['price']
                break
        if not order_flag and order_count<max_bid:
            ungain-=commision
            new_order={
                'id':timestep,
                'type':'sell',
                'price':new_close,
                'gain':0,
                'close':False
            }
    elif action==1:
        pass

    try:
        order.append(new_order)
    except:
        pass
    gaf = state_convert(data.iloc[timestep-window_size:timestep])
    model_fn = 'baseline_95.40__candlestick_500.pt'
    cnn = CNN().to(device)
    try:
        cnn.load_state_dict(torch.load(model_fn))
    except:
        cnn.load_state_dict(torch.load(model_fn,map_location=torch.device('cpu')))    
    cnn.eval()
    state=cnn(gaf)
    state=torch.cat((torch.tensor([long_short,order_count]).unsqueeze(0).to(device),state),axis=1)
    

    
    for ord in order:
        if ord['type']=='buy' and ord['close']:
            ungain+=ord['gain']

        elif ord['type']=='sell' and ord['close']:
            ungain+=ord['gain']

        elif ord['type']=='buy' and not ord['close']:
            ord['gain']=new_close-ord['price']
            ungain+=ord['gain']
            
        elif ord['type']=='sell'and not ord['close']:
            ord['gain']=ord['price']-new_close
            ungain+=ord['gain']


    try:
        if math.log(sortino)>0 and (ungain-last_ungain)>0:
            reward=(ungain-last_ungain)*(1+math.log(sortino))
        else:
            reward=(ungain-last_ungain)
    except:
        reward=(ungain-last_ungain)

    if timestep==len(data):
        done=True
    else:
        done=False
    
    return state, reward, order,done,long_short , order_count ,ungain