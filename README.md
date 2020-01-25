# Gait Phase Detection for Biped Robots  - Project description

dev/parser.ipynb - is not used<br>
directry run/ contains all executable scripts<br>
dev/ directory just for useful visualization

1. First of all we have to extract data for motion capture data to make it useful for analises.<br />
As a defult motion captured data has specific format where the human parameters described as sceleton and information of relative motion of each bone in sceleton. <br />
for this purpuses we prepeared script called amc_parser.py - here we extract the data from MOCAP (.amc(motion data) and .asf(sceleton information) data format) and express all data in cartesian space.

2. Export data to .csv in cartesian space



Original paper for citing <a href="https://www.researchgate.net/publication/336562551_Gait_Phase_Detection_for_Biped_Robots">I. Bzhikhatlov, K. Cheloshkina, M. Abramchuk "Gait Phase Detection for Biped Robots", 2019</a>.
