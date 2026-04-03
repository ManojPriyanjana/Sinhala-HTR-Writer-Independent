\# Thesis Results and Reproducibility Artifacts



This folder contains the final experiment outputs for the thesis:



\*\*A Writer-Independent Deep Learning Framework for Recognition of Sinhala Handwritten Text Lines from Lower Secondary Students\*\*



\## Purpose



These files are included as supporting research evidence for the final thesis experiment.  

They help supervisors, examiners, and other readers inspect the reported results, experiment configuration, writer-independent split evidence, and recorded training history.



This folder strengthens the academic credibility of the project by showing that the final results are supported by concrete output files rather than being reported only in descriptive form.



\## Experiment context



\- \*\*Task:\*\* Offline Sinhala handwritten text line recognition

\- \*\*Training environment:\*\* Google Colab

\- \*\*Evaluation protocol:\*\* Writer-independent split

\- \*\*Final model setting:\*\* DenseNet121-based CRNN with BiLSTM and CTC

\- \*\*Related notebook:\*\* `notebooks/training/final\_writer\_independent\_densenet121\_training.ipynb`



\## Final reported results



According to `final\_results.csv`, the final reported results are:



\- \*\*Validation CER:\*\* 0.066371

\- \*\*Validation WER:\*\* 0.207317

\- \*\*Test CER:\*\* 0.086424

\- \*\*Test WER:\*\* 0.251619



According to `run\_config.json`, the final evaluation used:



\- \*\*Decoder:\*\* beam search

\- \*\*Beam width:\*\* 10



\## Dataset split summary



According to `split\_summary.csv`, the final split used in this experiment was:



\- \*\*Training:\*\* 4411 samples, 276 writers

\- \*\*Validation:\*\* 555 samples, 34 writers

\- \*\*Test:\*\* 516 samples, 35 writers



According to `writer\_split.json`, the full count summary was:



\- \*\*Total samples:\*\* 5482

\- \*\*Total writers:\*\* 345



The overlap checks recorded in `writer\_split.json` were:



\- \*\*Train–Validation overlap:\*\* 0

\- \*\*Train–Test overlap:\*\* 0

\- \*\*Validation–Test overlap:\*\* 0



These records support the writer-independent evaluation claim made in the thesis.



\## Contents of this folder



\### `final\_results.csv`

Contains the final evaluation results of the selected experiment.



This file provides the main reported performance values for:

\- Character Error Rate (CER)

\- Word Error Rate (WER)

\- validation and test splits



\### `run\_config.json`

Contains the main experiment settings used in the final run.



This includes:

\- image height and width

\- batch size

\- train/validation/test ratios

\- number of epochs

\- learning rate

\- weight decay

\- patience

\- random seed

\- final decoder type

\- beam width

\- augmentation setting

\- normalization setting



\### `split\_summary.csv`

Provides a compact summary of the dataset split used for the final experiment.



This file is useful for quickly checking:

\- number of samples in each split

\- number of writers in each split

\- split proportions



\### `writer\_split.json`

Stores the detailed split metadata used in the experiment.



This file documents:

\- random seed

\- split ratios

\- sample and writer counts

\- overlap counts across splits

\- writer identifiers assigned to each split



It is one of the key reproducibility artifacts for the writer-independent setup.



\### `training\_history.csv`

Contains epoch-by-epoch training and validation records from the final run.



Its fields are:

\- `epoch`

\- `train\_loss`

\- `val\_cer`

\- `val\_wer`

\- `lr`



This file provides evidence that the model was trained and monitored over time.



\### `training\_curves.png`

Provides a visual summary of the training process.



This figure is useful for quickly inspecting:

\- training progression

\- convergence behaviour

\- validation trend across epochs



\### `final\_metrics\_table.png`

Provides an image version of the final results table.



This is useful for:

\- quick inspection

\- thesis discussion

\- viva presentation preparation



\### `test\_examples.csv`

Contains selected test examples with:

\- image path

\- ground-truth transcription

\- model prediction



This file supports qualitative review of the final model outputs.



\## Why these files matter



These artifacts are included to support:



\- \*\*Reproducibility\*\*  

&#x20; The configuration, split files, and recorded outputs make the final experiment easier to inspect and repeat.



\- \*\*Methodological transparency\*\*  

&#x20; The repository provides not only source code, but also the outputs that support the final reported results.



\- \*\*Writer-independent evaluation evidence\*\*  

&#x20; The split artifacts document that training, validation, and test writers were separated with zero overlap.



\- \*\*Result traceability\*\*  

&#x20; The reported thesis metrics can be traced back to concrete output files.



\## Suggested reading order



For anyone reviewing this repository, a useful order is:



1\. thesis document

2\. training notebook

3\. result artifacts in this folder

4\. demo application code



This gives a clear path from:

\- research problem

\- methodology

\- experiment outputs

\- practical demonstration



\## Important note



Only files that directly support the final experiment should be kept in this folder.  

System files such as `desktop.ini` are not research artifacts and should be removed from the repository.



\---



\*\*Author:\*\* Manoj Priyanjana  

\*\*Project:\*\* Sinhala HTR Writer-Independent Research Project

