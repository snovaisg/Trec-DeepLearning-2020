# Trec-DeepLearning-2020 

# Intro

This is an attempt at participating in the Deep Learning Track of Trec 2020 competition. The Task is passage ranking.

# Technologies used

- [pyterrier](https://github.com/terrier-org/pyterrier): A Python API for Terrier, which is an open source search engine.
- [pyterrier_bert](https://github.com/cmacdonald/pyterrier_bert): which includes two integrations of BERT. [CEDR](https://github.com/Georgetown-IR-Lab/cedr) and [BERT4IR]([BERT4IR](https://github.com/ArthurCamara/Bert4IR)).

# Scripts

It is recommended to run the python scripts as a background job. [tmux](https://github.com/tmux/tmux/wiki) helps accomplish that. Also you can log the output of the scripts in the following way: 

~~~bash
<myscript.py> >> <log_output_file.txt> 2>&1
~~~

Here's a rundown of all the scripts:

- **download_and_unzip_data**: bash script which downloads and unzips data into two folders:
  - *data folder*: contains data of 2020 deep learning passage track. Includes: queries, qrels and collection.
  - *data19 folder*: contains data of 2019 deep learning passage track. Includes: queries and qrels. (collection is the same as 2020 track).
- **create_index.py** Creates a terrier index of the track data (passages) and stores in the *index/* directory of the project. If the directory exists already, it will be overwritten. **Assumes** the passage data has already been downloaded and unzipped into the *data* directory.
- **cedr_bert4ir_baseline.py**: Creates two baselines using a pipeline that first retrieves passages using BM25 and then uses **cedr** or **bert4ir** respectively to rerank the passages. These two baselines are run for both the 2019 and 2020 topics. For 2019 we have the qrels eval so we just store the metrics map and ndcg of the retrieval just for comparisson with the 2019 leaderboard. For 2020 we store the retrieved passages for the eval topics, for later evaluation when the eval qrels become available. Saves results into the *results/* directory. 
- **pyterrier_MWE_trec_DL_2020.ipynb**: notebook that contains all the functionalities of the previous scripts in one place. (MWE from Minimal Working Example).

# Setup remote server

## Create google cloud compute instance

- [Webpage](https://console.cloud.google.com)
- Use 200 Gb storage ssd
- Choose the option with 8 CPUs and 30 GB RAM.

## Connections

### Normal Connection

~~~ bash
ssh -i <path_to_priv_key> <user>@<ip>
~~~

### jupyter notebook tunnel

~~~ bash
ssh -i <path_to_priv_key> -N -f -L localhost:<local_port_tunel>:localhost:<remote_port_where_notebook_running> <remote_ip>
~~~

### ratom connection

Edit files in remove server from atom in local computer. [webpage](https://atom.io/packages/remote-atom)

~~~bash
ssh -i <path_to_priv_key> -R 52698:localhost:52698 <user>@<ip>

~~~

## Install dependencies

~~~bash
sudo apt update
sudo apt install python3
sudo apt install wget
sudo apt-get install python3-distutils
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
sudo apt install git-all
sudo apt-get install python-dev gcc
sudo apt-get install gcc
sudo apt autoremove
sudo apt --fix-broken install
sudo apt-get install python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip
sudo apt-get install openjdk-11*
pip3 install ipywidgets
pip3 install --upgrade git+https://github.com/terrier-org/pyterrier.git#egg=python-terrier
pip3 install  --upgrade git+https://github.com/cmacdonald/pyterrier_bert.git
pip3 install jupyterlab
rm -rf get-pip.py
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
mv trec_eval /usr/local/bin/
~~~

**python version**: 3.7.3

## Additional setups

### Remote Edit from Atom in local computer

Atom package to browse and edit remote files using FTP and SFTP

[webpage](https://atom.io/packages/remote-edit)