

import torchvision as tv
import torch
import os

from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.Autoencoder_DNN import Model

criterion = torch.nn.MSELoss().to(device)
now = datetime.now()
NUM_EPOCHS = 60
BATCH_SIZE = 1000
SNR = 12
std = (2/10**SNR)**0.5

if __name__ == "__main__":
    for noise_location_idx in range(11,12):
        SAVE_PATH = "saved_models/autoencoder_" + now.strftime("%Y_%m_%d_%H_%M_%S")+"_"+str(noise_location_idx)

        if (not os.path.isdir("data")):
            os.mkdir("data/")
        if (not os.path.isdir("saved_models")):
            os.mkdir("saved_models/")

        training_data = tv.datasets.MNIST("data/", train=True, download=True, transform=tv.transforms.ToTensor())
        training_data = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
        test_data = tv.datasets.MNIST("data/", train=False, download=True, transform=tv.transforms.ToTensor())
        test_data = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

        net = Model(784).to(device)
        deepsc_optimizer = torch.optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-9)
        net.to(device)

        now = datetime.now()
        best_loss = torch.inf
        termination_count = 0
        noise_location_idx = 0

        # train
        for epoch in range(NUM_EPOCHS):
            net.train()

            # train
            total_loss = 0.

            for idx, (x_batch, target) in enumerate(training_data):
                x_batch = x_batch.to(device)
                target = target.to(device)
                inp_shape = x_batch.shape
                tmp = x_batch
                tmp = torch.flatten(tmp, 1)

                for _ in range(len(net.seq)) :
                    if _ == noise_location_idx:
                        sum_power = torch.sum(tmp ** 2, axis=1)
                        avg_power = sum_power / tmp.shape[1]
                        tmp = torch.einsum('ij,i->ij', tmp, 1 / avg_power ** 0.5)
                        tmp = tmp + torch.normal(0, std, tmp.shape).to(device)
                        tmp = torch.einsum('ij,i->ij', tmp, avg_power ** 0.5)
                    output = net.seq[_](tmp)
                    tmp = output

                output = output.reshape(x_batch.shape)
                loss = criterion(output,x_batch)

                deepsc_optimizer.zero_grad()
                loss.backward()
                deepsc_optimizer.step()
                # model_list[2].seq[0].weight[0]
                total_loss += loss.detach().item()
                mean_loss = total_loss / (idx + 1)


                print("\r" + now.strftime(
                    "%Y-%m-%d %H:%M:%S - ") +
                      'Epoch: {};  '
                      'Loss: {:.3f}; '
                      'batchs: {:d}/{:d}'.format
                      (epoch,
                       mean_loss,
                       idx + 1,
                       len(training_data)),
                      end="")
            print("\r", end="")
            print(
                "Epoch : " + str(epoch) +
                "; loss : %.2f" % (mean_loss))

            # test
            net.eval()
            with torch.no_grad():
                total_loss = 0.
                for idx, (x_batch, target) in enumerate(test_data):
                    x_batch = x_batch.to(device)
                    target = target.to(device)
                    inp_shape = x_batch.shape
                    tmp = x_batch
                    tmp = torch.flatten(tmp, 1)
                    for _ in range(len(net.seq)):
                        if _ == noise_location_idx:
                            sum_power = torch.sum(tmp ** 2, axis=1)
                            avg_power = sum_power / tmp.shape[1]
                            tmp = torch.einsum('ij,i->ij', tmp, 1 / avg_power ** 0.5)
                            tmp = tmp + torch.normal(0, std, tmp.shape).to(device)
                            tmp = torch.einsum('ij,i->ij', tmp, avg_power ** 0.5)
                        output = net.seq[_](tmp)
                        tmp = output
                    output = output.reshape(x_batch.shape)
                    loss = criterion(output, x_batch)

                    total_loss += loss.detach().item()
                    mean_loss = total_loss / (idx + 1)

                    print("\r" + now.strftime(
                        "%Y-%m-%d %H:%M:%S - ") +
                          'Epoch: {};  '
                          'Loss: {:.3f}; '
                          'batchs: {:d}/{:d}'.format
                          (epoch,
                           mean_loss,
                           idx + 1,
                           len(test_data)),
                          end="")
                print("\r", end="")
                print(
                    "Epoch : " + str(epoch) +
                    "; loss : %.2f" % (mean_loss))
                if best_loss>mean_loss :
                    best_loss = mean_loss
                    termination_count = 0
                    torch.save(net, SAVE_PATH)
                else :
                    termination_count +=1
                    net = torch.load(SAVE_PATH)
                    print("termination count :",termination_count)
                    if termination_count>= 5:
                        break