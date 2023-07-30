import torch
import matplotlib.pyplot as plt
import torchvision as tv


BATCH_SIZE = 1000
test_data = tv.datasets.MNIST("data/", train=False, download=True, transform=tv.transforms.ToTensor())
test_data = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

SAVE_PATH = "saved_models/autoencoder_2023_07_28_14_34_13_4"
test_scheme = "dnn"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss().to(device)

net = torch.load(SAVE_PATH).to(device)
noise_location_idx = 0
SNR = 12
std = (2/10**SNR)**0.5
for noise_location_idx in range(0,12):
    with torch.no_grad():
        ### test manifold
        for idx, (x_batch, target) in enumerate(test_data):
            x_batch = x_batch.to(device)
            target = target
            inp_shape = x_batch.shape
            tmp = x_batch
            tmp = torch.flatten(tmp, 1)

            for _ in range(5):  # 2 dimension result
                if _ == noise_location_idx:
                    sum_power = torch.sum(tmp**2,axis=1)
                    avg_power = sum_power/tmp.shape[1]
                    tmp= torch.einsum('ij,i->ij',tmp,1/avg_power**0.5)
                    tmp = tmp + torch.normal(0,std,tmp.shape).to(device)
                    tmp = torch.einsum('ij,i->ij', tmp, avg_power ** 0.5)

                output = net.seq[_](tmp)
                tmp = output

            output = output
            x_data = output.cpu().numpy()[:,0]
            y_data = output.cpu().numpy()[:,1]
            plt.scatter(x_data, y_data, c=target, s=3)

        plt.colorbar()
        #plt.show()
        plt.savefig("result_img/manifold_idx_"+str(noise_location_idx))
        plt.close('all')

        ### test re-generation
        n = 10
        plt.figure(figsize=(20*0.9, 4*0.9))
        for idx, (x_batch, target) in enumerate(test_data):
            x_batch = x_batch.to(device)
            inp_shape = x_batch.shape
            tmp = x_batch
            tmp = torch.flatten(tmp, 1)
            for _ in range(len(net.seq)):
                if _ == noise_location_idx:
                    tmp = tmp + torch.randn(tmp.shape).to(device)
                output = net.seq[_](tmp)
                tmp = output
            output = output.reshape(x_batch.shape)

            for i in range(n):
                ax = plt.subplot(2, n, i + 1)
                #plt.gray()
                ax.imshow(x_batch[i].reshape(28, 28).cpu())
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                ax = plt.subplot(2, n, i + 1 +  n)
                #plt.gray()
                ax.imshow(output[i].reshape(28, 28).cpu())
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_title('MSE : %.2f'%(criterion(output[i], x_batch[i]).item()))

            # plt.show()
            plt.savefig("result_img/result_idx_"+str(noise_location_idx))
            break
        plt.close('all')
