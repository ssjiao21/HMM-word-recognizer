For the primary system:

1. To train the parameters, run the command:
python -W ignore train_pri.py
This will use the files ‘clsp.endpts’, ’clsp.lblnames’, ‘clsp.trnlbls’, ‘clsp.trnscr’ as input, and generate the output files ‘baseform_pri.npy’, ‘eleHMM_pri.npy’, ‘silHMM_pri’, ‘dataProb_pri.csv’.

2. To test the system, run the command:
python -W ignore test_pri.py
This will use the files ’clsp.lblnames’, ’clsp.devlbls’, ‘baseform_pri.npy’, ‘eleHMM_pri.npy’, ‘silHMM_pri’, ‘dataProb_pri.csv’ as input, and generate the output file 'result_pri.txt’.

For the contrastive system:

3. To train the parameters, run the command:
python -W ignore train_cntra.py
This will use the file ‘clsp.endpts’, ’clsp.lblnames’, ‘clsp.trnlbls’, ‘clsp.trnscr’ as input, and generate the output files ‘baseform_cntra.npy’, ‘eleHMM_cntra.npy’, ‘silHMM_cntra’, ‘dataProb_cntra.csv’.

4. To test the system, run the command:
python -W ignore test_cntra.py
This will use the files ’clsp.lblnames’, ’clsp.devlbls’, ‘baseform_cntra.npy’, ‘eleHMM_cntra.npy’, ‘silHMM_cntra’, ‘dataProb_cntra.csv’ as input, and generate the output file 'result_cntra.txt’.


5. The two plots of the training data log-likelihood are drawn by running the command:
—-python plot.py
This will use the files ‘dataProb_pri.csv’ and ‘dataProb_cntra.csv’ as input, and generate the output files ‘plot_pri.png’ and ‘plot_cntra.png’