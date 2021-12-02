# Sequence-VAE
PyTorch Implementation of Generating Sentences from a Continuous Space by Bowman et al. 2015.

### Table of Contents

- **[Setup](#Setup)**
- [**Run the code**](#Run-the-code)
- **Training**
- **Inference**
- **Connect with me**
- **License** 

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

  <img src="./media/kl.jpg" style="zoom:3%;" />

- Reconstruction loss

  <img src="./media/reco.jpg" style="zoom:3%;" />

- KL loss vs Reconstruction loss

  <img src="./media/kl_reco.jpg" style="zoom:3%;" />

- ELBO loss

  <img src="./media/elbo.jpg" style="zoom:3%;" />





### Inference

#### 1. Sample Generation

Here is a generated sample from the model, when z ~ N(0,I) is given to the decoder.



#### 2. Interpolation

Here "President" word has been used as the start of the sentences. We randomly generated two sentences interpolated between them.

- Sentence 1: **President bush veto power changes meant to be a great number**

- Sentence 2: **President bush veto power opposed to the president of the house**

  

  **bush veto power opposed to the president of the house.**

```markdown
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president ' s council.
 bush veto power opposed to the president ' s council.
 bush veto power opposed to the president ' s council.
 bush veto power opposed to the president ' s council.
 bush veto power opposed to the president ' s council.
 bush veto power that kind of <unk> of natural gas.
 bush veto power changes to keep the <unk> and that.
 bush veto power changes to keep the <unk> and that.
 bush veto power changes that is in a telephone to.
 bush veto power changes that is in a telephone to.
 bush veto power changes meant to be a great number.
```

​	**bush veto power changes meant to be a great number.**

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

### Connect with me :slightly_smiling_face:

For any question or a collaboration, drop me a message [here](mailto:khamiesw@outlook.com?subject=[GitHub]%20Sequence-VAE%20Repo)

Follow me on [Linkedin](https://www.linkedin.com/in/khamiesw/)!

**Thank you :heart:**

### License 

![](https://img.shields.io/github/license/khamies/Sequence-VAE)