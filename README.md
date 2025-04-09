## Setup
### Env
Create the conda env
~~~
conda create --name MT_car_simulation python=3.8
conda activate MT_car_simulation
~~~
### Requirements
Install requirements
~~~
pip install -r requirements.txt
~~~
### OpenAI key
You need to create the file `keys/api_key.txt` and put your OpenAI key. Make sure to have acces to GPT4. 

## Run
To run a simulation with the Safe-NARRATE architecture.
~~~
python simulation.py
~~~
