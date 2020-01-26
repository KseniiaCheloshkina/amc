# Gait Phase Detection for Biped Robots

This project contains source code for the paper <a href="https://www.researchgate.net/publication/336562551_Gait_Phase_Detection_for_Biped_Robots">I. Bzhikhatlov, K. Cheloshkina, M. Abramchuk "Gait Phase Detection for Biped Robots", 2019</a>. The reference to source is obligatory if you use code or paper materials.

### Description
Gait   phase   detection   is   an important   task   for   walking cycle planning  because  the  stability  of  walking  could  be  estimated  by  gait  phase  sequence.  The  machine  learning  models  were  developed  for  detection  of  support  type  and  walking  gait phase  for  given  posture  to  be  used  for  stability  estimation  of  walking robots through phases sequence. It was revealed that it is possible  to  build  high  quality  model  based  on  human  motion  capture data using only normalized data from a body lower part without  toes  with  the  degrees  of  freedom  same  as  biped  robot  degrees  of  freedom.  Among  different  machine  learning  methods applied   to   support   type   prediction   by   all   available   motion   capture data gradient boosting model showed the highest quality in  7-fold  cross-validation  having 0.97  accuracy,  mean  per  class  precision   and   recall.   Additionally,   it   was   demonstrated   that removal of an upper half of body data as well as toes data did not lead  to  any significant  model  quality  decrease  as  a  final reduced model gives  0.964  accuracy,  mean  per  class  precision  and  recall.  Finally,  the  model  for  prediction  of  5  walking  gait  phases  was developed and showed comparable quality namely 0.95 accuracy, mean per class precision and recall.

### Structure of repository

  - /data - folder with all raw input data. Contains several folders with one folder per subject (.asf file and .amc file(s)).
  - /output - folder with preprocessed input data as well as results of analysis
  - /docs - folder with used documentation on data preprocessing
  - /run - contains all executable scripts
  
    amc_parser.py - script for transformation of raw motion capture data to suitable for analysis format (moving from local to global coordinates, coordinates normalization and saving to Pandas dataframe)<br />
    target.py - script for labeling gait phases<br />
    utils.py - script with useful functions <br />
  - /dev - folder with Jupyter notebooks for collection data, visualization of walking data, exploratory data analysis, model building and evaluation 

### Methods

1. Transformation of raw  motion capture data to suitable format.
First of all we have to extract data for motion capture data to make it useful for analysis.<br />
As a default motion captured data has specific format where the human parameters described as sceleton and information of relative motion of each bone in sceleton. <br />
For this purpuses we prepeared script called amc_parser.py - here we extract the data from MOCAP (.amc(motion data) and .asf(sceleton information) data format) and express all data in cartesian global space attached to "root". 
Export data to .csv in cartesian space

2. Labeling of gait phases with an algorithm (expert rules)

3. Analysis of gait cycle.

4. Prediction of support type with machine-learning models.

4. Prediction of gait phases with machine-learning models.
