# DigitRecognizer(MNIST)
Kaggle competion DigitRecognizer  모델 구현
 
최종 버전을 봐주세요

```python
class CNNModel(nn.Module):
        
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),            
            nn.AvgPool2d(2),
                
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),            
            nn.AvgPool2d(2),
               
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),            
            nn.AvgPool2d(2)
        )
            
        self.linear_layers = nn.Sequential(
            
            nn.Linear(1152, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            
            nn.Linear(64,64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
                
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            
            nn.Linear(32,32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
                
            nn.Linear(32, 10), 
            nn.LeakyReLU(0.1),
            nn.Softmax(dim=-1)  
        )
        
        self.dropout = nn.Dropout(0.25)
        self.flatten=nn.Flatten()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x=  self.flatten(x)
        x=  self.dropout(x)
        x = self.linear_layers(x)
        return x



```
CNN model로 MNIST 분류 진행!

Convolution layer + Pooling(Avgpool 사용)-> 2차원으로 압축(flatten) -> Linear layer 

->Soft max(다중분류에 많이 사용됨)을 통해 확률 구함


________________________________________________________________________________________

내 코드의 정확도가 30%가 나온 이유(최종 버전이 아닌 다른 버전)

Evalutaion 잘 못 만듬, val_batches를 사용하여 평가 해줬지만, 정확도를 val_batches가 아닌 test batches의 길이로 나눠 주었음.


