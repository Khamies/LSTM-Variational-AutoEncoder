# Sequence-VAE
PyTorch Implementation of Generating Sentences from a Continuous Space by Bowman et al. 2015.

### Table of Contents

### Setup

The code is using `pipenv` as a virtual environment and package manager. To run the code, all you need is to install the necessary dependencies. open the terminal and type:

- `git clone https://github.com/Khamies/Sequence-VAE.git` 
- `cd Sequence-VAE`
- `pipenv install`

And you should be ready to go to play with code and build upon it!

### Run the code

To train the model all you need is to type:

- `python main.py`

### Training

Here are the results from training the LSTM-VAE model:

- KL Loss

  <img src="./media/kl.jpg" style="zoom:5%;" />

- Reconstruction loss

  <img src="./media/reco.jpg" style="zoom:5%;" />

- KL loss vs Reconstruction loss

  <img src="./media/kl_reco.jpg" style="zoom:5%;" />

- ELBO loss

  <img src="./media/elbo.jpg" style="zoom:5%;" />

### Inference

-  

### Citation

> ```
> @misc{Khamies2021SequenceVAE,
>   author = {Khamies, Waleed},
>   title = {PyTorch Implementation of Generating Sentences from a Continuous Space by Bowman et al. 2015},
>   year = {2021},
>   publisher = {GitHub},
>   journal = {GitHub repository},
>   howpublished = {\url{https://github.com/Khamies/Sequence-VAE}},
> }
> ```

### Connect



### License