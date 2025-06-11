# Wordsworth
Wordsworth: A generative word dataset for comparison of speech representations in humans and neural networks.    
Please refer https://osf.io/pu7f2/ for datasets and models

## WW_cochleagram_generator.py
WW_cochleagram_generator.py will generate cochleagrams for all word waveform tokens. Make sure PyTorch and cochleagram master were pre-installed (https://github.com/jenellefeather/chcochleagram). To get the correct waveform input and cochleagram output, please modify the waveform pathway by python WW_cochleagram_generator.py --full_path='root_path/Wordsworth_v1.0/'. 

Please cite the paper below for cochleagram generator: 

[Feather, J., Leclerc, G., Mądry, A., & McDermott, J. H. (2022). Model metamers illuminate divergences between biological and artificial neural networks. bioRxiv.](https://github.com/jenellefeather/chcochleagram#:~:text=Feather%2C%20J.%2C%20Leclerc%2C%20G.%2C%20M%C4%85dry%2C%20A.%2C%20%26%20McDermott%2C%20J.%20H.%20(2022).%20Model%20metamers%20illuminate%20divergences%20between%20biological%20and%20artificial%20neural%20networks.%20bioRxiv.)

## word_generator_package.py
The word_generator_package is a self-built library based on the Google Cloud API for loading generative models and generating waveforms. Please make sure to acquire Google Cloud access and set up service account key/credentials. For credentials setting up details, please see https://developers.google.com/workspace/guides/create-credentials. 

## wordsworth_dataloader.py
The word_generator_package is a standard PyTorch dataloader package. The WaveDataLoader class is used for setting up the waveform dataset by input root path, and for batch loading the data using the PyTorch DataLoader. The CochleagramDataLoader class is used for setting up the cochleagram dataset (please make sure cochleagram were pre-generated). Note that the root path is the local path that includes all the words (ant, ape, badge and etc.). The function generate_WW_subset can be used to generate a subset of words with certain criteria. For example, users could choose a particular set of desired features (e.g.,e.g. speaking rate, accent, etc) and acquire all the tokens that fit the criteria from the full set. See demo.py for detailed usage. 
