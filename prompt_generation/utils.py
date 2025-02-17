import os
import logging


def set_path(save_root='result', save_tag=""):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    if(save_tag == ""):
        save_tag = f"llm_graph_of_toughts"
    exp_seq_path = os.path.join(save_root, 'exp_seq.txt')

    if not os.path.exists(exp_seq_path):
        file = open(exp_seq_path, 'w')
        exp_seq=0
        exp_seq = str(exp_seq)
        file.write(exp_seq)
        file.close
        save_tag = 'exp_' + exp_seq + '_' + save_tag
    else:
        file = open(exp_seq_path, 'r')
        exp_seq = int(file.read())
        exp_seq += 1
        exp_seq = str(exp_seq)
        save_tag = 'exp_' + exp_seq + '_' + save_tag
        file = open(exp_seq_path, 'w')
        file.write(exp_seq)
        file.close()

    exp_seq = exp_seq
    save_path = os.path.join(save_root, save_tag)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config_path = os.path.join(save_path, 'config.json')
    logger_path = os.path.join(save_path, 'exp_log.log')    


    return logger_path, exp_seq, save_path


def get_logger(save_root='result', save_tag=""):

    logger_path, exp_seq, save_path = set_path(save_root=save_root, save_tag=save_tag)

    logging.basicConfig(
        filename=logger_path,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M', 
        level=logging.DEBUG, 
        filemode='w'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    return logger, exp_seq, save_path