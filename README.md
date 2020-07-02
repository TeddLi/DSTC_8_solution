# DSTC_8_solution
This is our proposed model for subtask 1 and 2 of Dialog System Technology Challenges 8 (DSTC 8)
### Dependencies
Python 3.6
Tensorflow 1.4.0

### Quick start

##### Fill the dataset and BERT model according to filename.
##### Adaptation process
Open DSTC_8_solution/Pre_trainprocess/sh_file
Run bash REBED_advisor_adapt_external.sh  and  bash REBED_ubuntu_adapt_external.sh
for advisor dataset and ubuntu dataset, respectively.

##### Finetuen process
Open DSTC_8_solution/DSTC_finetune/sh_file
Train scripts run_advisor_HAE.sh, run_ubuntu_HAE.sh, run_task2_REBED_adapt.sh
Eval scrips eval_task1_advisor.sh, eval_task1_ubuntu.sh, eval_task2.sh



### Cite
If you use the code and datasets, please cite the following paper:

@misc{gu2020pretrained,
    title={Pre-Trained and Attention-Based Neural Networks for Building Noetic Task-Oriented Dialogue Systems},
    author={Jia-Chen Gu and Tianda Li and Quan Liu and Xiaodan Zhu and Zhen-Hua Ling and Yu-Ping Ruan},
    year={2020},
    eprint={2004.01940},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
      
