# Jobify

We're developing a way for managers and professionals to find what they need.

> Jobify: a conversational agent for job-related search; 
Human Machine Dialogue course 2024. 
> 
> [[Video-DEMO]](https://drive.google.com/file/d/1J6thJvAPIvKIOYXMmLqJPIDsP51QQjxv/view?usp=sharing), [[REPORT]](https://drive.google.com/file/d/1CrbSDHgzTDosYDnXOu5yaBsfPevm41N4/view?usp=sharing)
>
> Authors: [Andrea Coppari](https://it.linkedin.com/in/andreacoppari1005), [Riccardo Tedoldi](https://www.linkedin.com/in/riccardo-tedoldi-460269291/)
> 
> Supervisors: [Giuliano Tortoreto](https://www.linkedin.com/in/giuliano-tortoreto/?locale=it_IT), [S. Mahed Mousavi](https://scholar.google.com/citations?user=OYqE3uAAAAAJ&hl=en)
> 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

Due to the nature of the project, we employed LLMs so it may require a lot of computational power to run the project.

### Prerequisites

What things you need to install the software and how to install them.

### Installing

You need to clone the repository and enter in the src folder. Then, to install the necessary packages from requirements.txt into a conda virtual environment.

```bash
# Create the virtual environment
conda create -n jobify --file requirements.txt

# Activate the virtual environment
conda activate jobify
```

or

Run the makefile to install all the necessary packages and vectorize the DBs.

```bash
make
```

To vectorize the DBs manually, there is the following script `src\installation\install.py`. If you have issues with bitsandbytes, you can checkout the first cell of the notebook `notebooks\02_mistralLLM\minstral-llm.ipynb`.

## Running the tests

To run the rasa train:

```bash
rasa train
```

To run the rasa shell:

```bash
rasa shell -m <path_to_model>
```

To run the rasa server:

```bash
rasa run
```

To run the rasa actions server:

```bash
rasa run actions
```

To run the rasa interactive shell:

```bash
rasa interactive
```

To run the rasa test:

```bash
rasa test nlu --cross-validation --fold 10
rasa test core 
```

## Alexa skill configuration

To run the Alexa skill, you need to create a new skill in the Alexa Developer Console and configure the endpoint to the rasa server. We suggest to use ngrok to create a secure tunnel to the rasa server. Here is a guide on how to do it: [youtube playlist guide](https://www.youtube.com/watch?v=zkM6jORrark).


## Structure of the repository

The repository is structured as follows:

```
* notebooks
    * 00_userProfilesGeneration
    * 02_semanticSearch
    * 03_mistralLLM
    * data
    * utils
* src
    * actions
        * actions.py
    * data
        * nlu.yml
        * stories.yml
        * rules.yml
    * models
    * installation
        * data
        * install.py
        * vector_db.py
    * tests
        * test_stories.yml
    * results
    * config.yml
    * credentials.yml
    * domain.yml
    * endpoints.yml
    * Makefile
    * README.md
    ...
    * requirements.txt
```


## Built With

* [Rasa](https://rasa.com/) - The chatbot framework used
* [Anaconda](https://www.anaconda.com/) - The data science platform used
* [Alexa](https://developer.amazon.com/alexa) - The voice assistant used
* [LangChain](https://langchain.com/) - The language processing platform used
* [ngrok](https://ngrok.com/) - The secure tunnel used
* [Hugging Face](https://huggingface.co/) - The transformer models used

## Contributing

If you want to contribute to the project, please contact us.

* **Riccardo Tedoldi** - [LinkedIn](https://www.linkedin.com/in/riccardo-tedoldi-460269291/)
* **Andrea Coppari** - [LinkedIn](https://it.linkedin.com/in/andreacoppari1005)

We are glad to accept any kind of contribution.

## To Cite

```bibtex
@misc{CoppariTedoldi2023,
    title   = {Jobify: a conversational agent for job-related search},
    author  = {Andrea Coppari, Riccardo Tedoldi},
    year    = {2024},
    url  = {https://github.com/r1cc4r2o/Jobify}
}
```