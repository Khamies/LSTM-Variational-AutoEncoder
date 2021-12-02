from data.ptb import PTB

import torch
from loss import VAE_Loss
from model import LSTM_VAE
from train import Trainer

from settings import global_setting, model_setting, training_setting

from utils import  interpolate, plot_elbo, get_latent_codes, visualize_latent_codes

# General Settings

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(global_setting["seed"])

# Load the data
train_data = PTB(data_dir="./data", split="train", create_data= False, max_sequence_length= training_setting["bptt"])
test_data = PTB(data_dir="./data", split="test", create_data= False, max_sequence_length= training_setting["bptt"])
valid_data = PTB(data_dir="./data", split="valid", create_data= False, max_sequence_length= training_setting["bptt"])

# Batchify the data
train_loader = torch.utils.data.DataLoader( dataset= train_data, batch_size= training_setting["batch_size"], shuffle= True)
test_loader = torch.utils.data.DataLoader( dataset= test_data, batch_size= training_setting["batch_size"], shuffle= True)
valid_loader = torch.utils.data.DataLoader( dataset= valid_data, batch_size= training_setting["batch_size"], shuffle= True)



vocab_size = train_data.vocab_size
model = LSTM_VAE(vocab_size = vocab_size, embed_size = model_setting["embed_size"], hidden_size = model_setting["hidden_size"], latent_size = model_setting["latent_size"]).to(device)

Loss = VAE_Loss()
optimizer = torch.optim.Adam(model.parameters(), lr= training_setting["lr"])

trainer = Trainer(train_loader, test_loader, model, Loss, optimizer)




if __name__ == "__main__":

    # Epochs
    train_losses = []
    test_losses = []
    for epoch in range(training_setting["epochs"]):
        print("Epoch: ", epoch)
        print("Training.......")
        train_losses = trainer.train(train_losses, epoch, training_setting["batch_size"], training_setting["clip"])
        print("Testing.......")
        test_losses = trainer.test(test_losses, epoch, training_setting["batch_size"])


    plot_elbo(train_losses, "train")
    plot_elbo(test_losses, "test")




