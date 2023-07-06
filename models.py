import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from all_functions import *

transform = transforms.Compose(
    [
        transforms.Resize((128,128)),
#         transforms.ToTensor()
    ]
)
def transform_image_to_quazi(batch):
    transform = transforms.Compose(
        [
            transforms.Resize((128,128)),
        ]
    )
    
    resolution = batch[1].shape[-2]
    test = batch[1].reshape((-1,3,resolution,resolution))
    test = transform(test)
    return test

def transform_image_to_norm_quazi(batch):
    transform = transforms.Compose(
        [
            transforms.Resize((128,128)),
        ]
    )
    
    resolution = batch[1].shape[-2]
    test = batch[1].reshape((-1,3,resolution,resolution))
    test = transform(test)
    return np.array(test)



class quasi_conv(nn.Module):
    def __init__(self,sizes):
        
        super(quasi_conv, self).__init__()
        
        self.flatten = nn.Flatten(start_dim = 2, end_dim = -1)
                
        self.permute = lambda x: x.permute(0, 2, 1)
        
        self.relu = nn.LeakyReLU()
        
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])
        
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2)
        
        self.out_dim = sizes[2]
    
    def fit(self, inputs):
#         print('-------')
        output = self.flatten(inputs)
        output = output.permute(0, 2, 1)
#         print(f'{output.shape}')
        output = self.relu(self.fc1(output))
#         print(f'{output.shape}')
        output = self.relu(self.fc2(output))
#         print(f'{output.shape}')
        output = output.permute(0, 2, 1)
        #output = output.resize(inputs.shape[0], self.out_dim, inputs.shape[-2], inputs.shape[-1])
        output = output.view(inputs.shape[0], self.out_dim, inputs.shape[-2], inputs.shape[-1])
        output = F.interpolate(output, size=(inputs.shape[-2], inputs.shape[-1]), mode='bilinear', align_corners=True)




#         print(f'{output.shape}')
        output = self.max_pooling1(output)
#         print(f'{output.shape}')
#         print('-------')
        return output

class quasy_yolo(nn.Module):
    def __init__(self):
        super(quasy_yolo, self).__init__()

        self.train_losses = []
        self.test_losses = []

        
        self.first_conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1)
        self.first_conv2 = nn.Conv2d(in_channels=20, out_channels=48, kernel_size=3, padding=1)
        
        self.first_pool1 = nn.MaxPool2d(kernel_size=2)
        self.first_pool2 = nn.MaxPool2d(kernel_size=2)
        
        
        self.fc1 = nn.Linear(48, 64)
        self.fc2 = nn.Linear(64, 128)
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2)
        
        self.fc3 = nn.Linear(128, 56)
        self.fc4 = nn.Linear(56, 5)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2)
        
        self.convs = nn.ModuleList([quasi_conv([48,64,128]), 
                                    quasi_conv([128,56,5])])
        
        
        self.flatten = nn.Flatten(start_dim = 2, end_dim = -1)
        self.relu = nn.LeakyReLU()

        
    def forward(self, image):
        image = image.float()
#         print('Start Dimension: ', image.shape)
        output = self.first_conv1(image)
        output = self.first_pool1(output)
        output = self.first_conv2(output)
        output = self.first_pool2(output)
        
#         print('After first conv: ', output.shape)
        
        output = self.convs[0].fit(output)
        
#         print('First layer: ', output.shape)
        
        output = self.convs[1].fit(output)
        
#         print('Second layer: ', output.shape)

        return output


    def fit(self, loader_train_discriminator, loader_test_discriminator, criterion = nn.MSELoss(reduction='mean'), num_epoch = 25000):
        torch.manual_seed(101)
#         criterion = nn.MSELoss(reduction='mean')
#         optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        transform = transforms.Compose(
            [
                transforms.Resize((128,128)),
        #         transforms.ToTensor()
            ]
        )
        
#         num_epoch = 25000

        for epoch in range(num_epoch):
        #     indexes_shuffle = list(range(len(answer_tensor_train)))
        #     random.shuffle(indexes_shuffle)
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


def get_tensor_quazy(path, model):
    
    transform = transforms.Compose(
            [
                transforms.Resize((128,128)),
        #         transforms.ToTensor()
            ]
    )
    
    image = cv2.imread(path)
    image = torch.Tensor([image])/255.0
    resolution = image.shape[-2]
    
    test = image.reshape((-1,3,resolution,resolution))
    
    test = transform(test)

    res = model(test)
    
    return res

def detect_objects_quazy(path, model, accept_q):
    
    transform = transforms.Compose(
            [
                transforms.Resize((128,128)),
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
#     print(tensor.shape)
    tensor_border=border(tensor,accept_q)
    im=image[0,:,:,:].numpy()
#     print(im.shape)
    visual(tensor_border, (im*255).astype(np.uint8))
#     ai_2(im[ :,:,0])


class usual_conv(nn.Module):
    def __init__(self):
        super(usual_conv, self).__init__()
        
        self.train_losses = []
        self.test_losses = []
        
        self.first_conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1)
        self.first_conv2 = nn.Conv2d(in_channels=20, out_channels=48, kernel_size=3, padding=1)
        
        self.first_pool1 = nn.MaxPool2d(kernel_size=2)
        self.first_pool2 = nn.MaxPool2d(kernel_size=2)
        
        
        self.fc1 = nn.Linear(48, 64)
        self.fc2 = nn.Linear(64, 128)
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2)
        
        self.fc3 = nn.Linear(128, 56)
        self.fc4 = nn.Linear(56, 5)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv1 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=5, kernel_size=3, padding=1)
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2)
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2)
        
#         self.convs = nn.ModuleList([quasi_conv([48,64,128]), 
#                                     quasi_conv([128,56,5])])
#         self.quasi1 = quasi_conv([48,64,128])
        
        
        # ???? [128, 64, 5]
#         self.quasi2 = quasi_conv([128,56,5])
        # ???? [128, 64, 5]
        
        
        self.flatten = nn.Flatten(start_dim = 2, end_dim = -1)
        self.relu = nn.LeakyReLU()

        
    def forward(self, image):
        image = image.float()
#         print('Start Dimension: ', image.shape)
        output = self.first_conv1(image)
        output = self.first_pool1(output)
        output = self.first_conv2(output)
        output = self.first_pool2(output)
        
#         print('After first conv: ', output.shape)
        
        output = self.conv1(output)
        output = self.max_pooling3(output)
#         print('First layer: ', output.shape)
        
        output = self.conv2(output)
        output = self.max_pooling4(output)
#         print('Second layer: ', output.shape)

        return output

    def fit(self, loader_train_discriminator, loader_test_discriminator, criterion = nn.MSELoss(reduction='mean'), num_epoch = 25000):
        torch.manual_seed(101)
#         criterion = nn.MSELoss(reduction='mean')
#         optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        transform = transforms.Compose(
            [
                transforms.Resize((128,128)),
        #         transforms.ToTensor()
            ]
        )
        
#         num_epoch = 25000

        for epoch in range(num_epoch):
        #     indexes_shuffle = list(range(len(answer_tensor_train)))
        #     random.shuffle(indexes_shuffle)
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


def get_tensor_conv(path, model):
    
    transform = transforms.Compose(
            [
                transforms.Resize((128,128)),
        #         transforms.ToTensor()
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
                transforms.Resize((128,128)),
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
#     print(tensor.shape)
    tensor_border=border(tensor,accept_q)
    im=image[0,:,:,:].numpy()
#     print(im.shape)
    visual(tensor_border, (im*255).astype(np.uint8))
#     ai_2(im[ :,:,0])

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
    return (accuracy, precision, recall, f1)

def IoU_norm(loader_discriminator, model, border_accept, border):
    ex = next(iter(loader_discriminator))
    resolution = ex[1].shape[-2]
    test = ex[1].reshape((-1,3,resolution,resolution))
    test = transform(test)

    res = model(np.array(test))
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
                TN += 1
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    precision = TP/(TP+FP+0.00001)
    recall = TP/(TP+FN+0.00001)
    f1 = 2 * precision*recall/(precision+recall+0.00001)
    return (accuracy, precision, recall, f1)
