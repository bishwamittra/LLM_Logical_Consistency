A repository to assess and improve the logical consistency of large language models

## Install Libraries
- Tested on `Python 3.8.0`
  ```
  pip install -r requirements.in
  pip install -r requirements.txt
  ```

## Download (Processed) Data
Download and extract in the root: [Link](https://drive.google.com/file/d/19TnpWma7Ht3J_N50ZohRjnLh5dQtNxuk/view?usp=sharing)

For reproducibility of dataset preparation, the steps are the following.


### 1. KG Dataset
- Clone Query2Box repo to get dataset FB15k and NELL
    ```
    git clone https://github.com/hyren/query2box.git
    ```
   There should be a folder `query2box/data` in the root.

- Get entity map file for FB15k from [Dropbox](https://www.dropbox.com/scl/fi/z0z935ijqcg39xcdnk0mu/entity.txt?rlkey=yido61oxg9lpuqyuwnktbqqwh&st=hzkizkss&dl=0) and place it under `query2box/data/`.
- To prepare entity map file for NELL, run the following commands.
    ```
    cd data_generation
    python dataset_prepare_NELL.py
    ```

- To get  WikiKG90Mv2 dataset, run following commands.

    ```
    cd data_generation
    python data_prepare_wiki.py
    ```

    Dataset is stored inside `query2box/data`.



### 2. Logic Fact Checking Dataset Preparation
- Data process. 
    - Step 1: map entity and relation id to names. 
    - Step 2: Context creation by running BFS and find flipped entity.
    ```
    cd data_generation
    bash todo_data_preprocess.sh
    bash todo_data_generation.sh
    ```

- Prompt generation
    ```
    cd prompt_generation
    bash todo.sh
    cp result/exp*/*.csv ../data_optimized/prompts 



## LLMs
- Get the LLMs from hugginface and store locally
    ```
    cd fine-tuning
    python download_model.py
    ```



## Consistency Assessment
- Before fine-tuning
    ```
    cd consistency_check
    bash todo.sh
    ```

- After fine-tuning
    ```
    cd consistency_check
    bash todo_checkpoint.sh
    ```

## Fine-tuning
- Update wandb api key in `fine-tuning/finetune.py`
- For fine-tuning run following commandas
    ```
    cd fine-tuning
    bash todo_finetune.sh
    ```

## Visualize Results
- Results are stored in `consistency_check/result*` directory. Run the following command from the root.
    ```
    bash read_output.sh
    ```


