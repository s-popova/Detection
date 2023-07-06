import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2

from all_functions import *

transform = transforms.Compose(
    [
        transforms.Resize((128,128), antialias=True),
    ]
)
def transform_image(batch):
    transform = transforms.Compose(
        [
            transforms.Resize((128,128), antialias=True),
        ]
    )
    
    resolution = batch[1].shape[-2]
    print(resolution.shape)
    test = batch[1].reshape((-1,3,resolution,resolution))
    test = transform(test)
    return test

def transform_image_to_norm(batch):
    transform = transforms.Compose(
        [
            transforms.Resize((128,128), antialias=True),
        ]
    )
    
    resolution = batch[1].shape[-2]
    test = batch[1].reshape((-1,3,resolution,resolution))
    test = transform(test)
    return np.array(test)

class pyramid_conv(nn.Module):
    def __init__(self, num_filteres_in=3, num_filteres_out=1,num_filteres_middle=1):
        """
        На вход подаются следующие параметры:
         :param num_filteres_in - размер канальности тензора на вход
         :param num_filteres_out - размер канальности тензора на выход
         :param num_filteres_middle - промежуточный размер канальности тензора
         :param L1 и L2 - параметры для регуляризации
         :param device - Выбор CPU/GPU(:cuda)
        """
        super( pyramid_conv , self).__init__()
        
        p_z=0 
        k_z=1
        bias_=1
        
        self.pconv1 = nn.Conv2d(in_channels=num_filteres_in, out_channels=num_filteres_middle, kernel_size=3, stride=1, padding=2, dilation=2)
        self.pconv2 = nn.Conv2d(in_channels=num_filteres_in, out_channels=num_filteres_middle, kernel_size=3, stride=1, padding=4, dilation=4)
        self.pconv3 = nn.Conv2d(in_channels=num_filteres_in, out_channels=num_filteres_middle, kernel_size=3, stride=1, padding=8, dilation=8)
        self.pconv4 = nn.Conv2d(in_channels=num_filteres_in, out_channels=num_filteres_middle, kernel_size=3, stride=1, padding=12, dilation=12)
        
        self.bn1 = nn.BatchNorm2d(num_filteres_middle)
        self.bn2 = nn.BatchNorm2d(num_filteres_middle)
        self.bn3 = nn.BatchNorm2d(num_filteres_middle)
        self.bn4 = nn.BatchNorm2d(num_filteres_middle)
        
        self.layer_conv_31 = nn.Conv2d(in_channels=4*num_filteres_middle, 
                                       out_channels=8*num_filteres_middle, 
                                       kernel_size=(k_z, k_z), 
                                       stride=(1, 1), 
                                       padding=(p_z, p_z), 
                                       padding_mode='zeros', 
                                       bias=bias_)
        self.layer_activation_31 = nn.LeakyReLU(0.05)  
        self.layer_conv_41 = nn.Conv2d(in_channels=8*num_filteres_middle, 
                                       out_channels=num_filteres_out, 
                                       kernel_size=(k_z, k_z),
                                       stride=(1, 1), 
                                       padding=(p_z, p_z), 
                                       padding_mode='zeros', 
                                       bias=bias_)
        self.layer_batch_norm_3 = nn.BatchNorm2d(num_features=num_filteres_out)
        self.layer_activation_41 = nn.LeakyReLU(0.05)  

  
        self.relu = nn.ReLU()

    def fit(self, x):
        # apply each convolutional layer in the atrous pyramid
          
        x1 = self.pconv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.pconv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.pconv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        x4 = self.pconv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)

        # concatenate the outputs from each convolutional layer
        out = torch.cat([x1, x2, x3, x4], dim=1)
        
        x = self.layer_conv_31(out)
        x = self.layer_activation_31(x)
        x = self.layer_conv_41(x)
        x = self.layer_batch_norm_3(x)
        x = self.layer_activation_41(x)
        
        return x

class pyramid_yolo(nn.Module):
    def __init__(self):
        super(pyramid_yolo, self).__init__()

        self.train_losses = []
        self.test_losses = []

        
        self.first_conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, padding=1)
        
        self.convs = pyramid_conv(num_filteres_in = 15,num_filteres_middle = 18,num_filteres_out=20)
        
        self.first_conv2 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, padding=1)
        
        self.convs0 = pyramid_conv(num_filteres_in = 32,num_filteres_middle = 40,num_filteres_out=48)
        
        
        self.first_pool1 = nn.MaxPool2d(kernel_size=2)
        self.first_pool2 = nn.MaxPool2d(kernel_size=2)
        
        
        self.fc1 = nn.Linear(48, 64)
        self.fc2 = nn.Linear(64, 128)
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2)
        
        self.fc3 = nn.Linear(128, 56)
        self.fc4 = nn.Linear(56, 5)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2)
        
        self.convs1 = pyramid_conv(num_filteres_in = 48,num_filteres_middle = 64,num_filteres_out=128)        
        self.convs2 = pyramid_conv(num_filteres_in = 128,num_filteres_middle = 56,num_filteres_out=5)
        
        
        
        self.flatten = nn.Flatten(start_dim = 2, end_dim = -1)
        self.relu = nn.LeakyReLU()

        
    def forward(self, image):
        image = image.float()
        #print('Start Dimension: ', image.shape)
        output = self.first_conv1(image)
        output = self.first_pool1(output)
        output = self.convs.fit(output)
        output = self.first_conv2(output)
        output = self.first_pool2(output)
        output = self.convs0.fit(output)
        #print('After first conv: ', output.shape)
        output = self.convs1.fit(output)
        #print('First layer: ', output.shape)
        output = self.convs2.fit(output)
        #print(output.shape)
        output = torch.nn.functional.interpolate(output, size=(8, 8), mode='bilinear', align_corners=False)

        return output


    def fit(self, loader_train_discriminator, loader_test_discriminator, criterion = nn.MSELoss(reduction='mean'), num_epoch = 25000):
        torch.manual_seed(101)

        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        transform = transforms.Compose(
            [
                transforms.Resize((128,128), antialias=True),
            ]
        )
        

        for epoch in range(num_epoch):

            train_error = 0
            test_error = 0
            for i, s in enumerate(loader_train_discriminator):

                resolution = s[1].shape[-2]
                test = s[1].reshape((-1,3,resolution,resolution))
                test = transform(test)

                self.zero_grad()

                target = s[0]

                outputs = self(test)
                outputs = outputs.permute(0, 2, 3, 1)

                target = target.float()
                outputs = outputs.float()
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                
                ######################################33
                
                for i_test, s_test in enumerate(loader_test_discriminator):
                    resolution = s_test[1].shape[-2]
                    test_test = s_test[1].reshape((-1,3,resolution,resolution))
                    test_test = transform(test_test)

                    self.zero_grad()

                    target_test = s_test[0]

                    outputs_test = self(test_test)
                    outputs_test = outputs_test.permute(0, 2, 3, 1)

                    target_test = target_test.float()
                    outputs_test = outputs_test.float()
                    loss_test = criterion(outputs_test, target_test)

                
                ##########################################
                
                print(f'Stage {i + 1}/{len(loader_train_discriminator)}, train: {loss.item()}, test: {loss_test.item()}')
                train_error += loss.item()
                test_error += loss_test.item()

            print('-----------------')
            print(f'Epoch {epoch + 1}/{num_epoch}, Loss: {test_error}')
            self.train_losses.append(train_error)
            self.test_losses.append(test_error)
            print('-----------------')
            if epoch+1 % 100 == 0:
                plt.figure(figsize=(25, 25))
                plt.plot(range(len(test_losses)),test_losses)
                plt.show()


def get_tensor(path, model):
    
    transform = transforms.Compose(
            [
                transforms.Resize((128,128), antialias=True),  
            ]
    )
    
    image = cv2.imread(path)
    image = torch.Tensor([image])/255.0
    resolution = image.shape[-2]
    
    test = image.reshape((-1,3,resolution,resolution))
    
    test = transform(test)

    res = model(test)
    
    return res

def detect_objects_conv(path, model, accept_q):
    
    transform = transforms.Compose(
            [
                transforms.Resize((128,128), antialias=True),
        #         transforms.ToTensor()
            ]
    )
    
    image = cv2.imread(path)
    image = torch.Tensor([image])/255.0
    resolution = image.shape[-2]
    
    test = image.reshape((-1,3,resolution,resolution))
    
    test = transform(test)

    res = model(test)

    res = res.permute(0, 2, 3, 1)

    tensor=res[0,:,:,:].detach().numpy()

    tensor_border=border(tensor,accept_q)
    im=image[0,:,:,:].numpy()

    visual(tensor_border, (im*255).astype(np.uint8))


def IoU(loader_discriminator, model, border_accept, border):
    ex = next(iter(loader_discriminator))
    resolution = ex[1].shape[-2]
    test = ex[1].reshape((-1,3,resolution,resolution))
    test = transform(test)

    res = model(test)
    res = res.permute(0, 2, 3, 1)

    results = res[0]
    answers = ex[0][0]
    
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(len(results)):
        for j in range(len(results)):
            [x_res, y_res, w_res, h_res, p_res] = results[i][j]
            [x_ans, y_ans, w_ans, h_ans, p_ans] = answers[i][j]
            
            if p_res >= border and p_ans < border_accept:
                FP += 1
            elif p_ans < border and p_res >= border_accept:
                FN += 1
            elif p_ans >= border and p_res >= border_accept:
                width = abs(max(x_res - w_res/2, x_ans - w_ans/2) - min(x_res + w_res/2, x_ans + w_ans/2))
                height = abs(max(y_res - h_res/2, y_ans - h_ans/2) - min(y_res + h_res/2, y_ans + h_ans/2))
                inter = width * height
                un = w_res * h_res + w_ans * h_ans - inter
                if inter/un > border:
                    TP += 1
                else:
                    FP += 1
            else:
                TN += 1
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    if TP+FP == 0:
        precision = TP/(TP+FP+0.00001)
    else:
        precision = TP/(TP+FP)
    if TP+FN == 0:
        recall = TP/(TP+FN+0.00001)
    else:
        recall = TP/(TP+FN)
    if precision+recall == 0:
        f1 = 2 * precision*recall/(precision+recall+0.00001)
    else:
        f1 = 2 * precision*recall/(precision+recall)
    return (' accuracy = ', accuracy, ' precision = ',  precision, ' recall = ',recall, ' f1 = ' , f1)




