# Constrained-BO

**Project Leader:** Aleks Siemenn \<<asiemenn@mit.edu>\>

**Collaborators:** Felipe Oviedo, Armi Tiihonen, Tonio Buonassisi

**Abstract:** Bayesian optimization (BO) is a statistical tool that has beenwidely implemented to optimize complex physical systemswhere data volume is low, e.g., additively manufactured structures and perovskite solar cells. BO samples optimized points from the parameter space of the physical system using an acquisition function – a probabilistic relationship between the means and the variances of thetarget objective. However, this acquisition function does notintrinsically utilize physical information to help guide thesampling of these optimized points. In this paper, we use physical data from our system that contains latent information about the target objective to constrain the acquisition function to more optimal regions of the parameter space.The rate of optimized objective discovery of the proposed constrained BO method is compared to that of traditional unconstrained BO.

**Github Repo:** \<<https://github.com/PV-Lab/Constrained-BO>\>

**Location of data:**

[1] Internally available: Dropbox (MIT)\Buonassisi-Group\ASD Team\Archerfish\05_Data\Imaged_droplets

**Sponsors:** C2C

*******

## Explanation of code within GitHub Repo:

### [1] Constraint_function.ipynb
Performs constrained Bayesian optimization. Physical information (number of droplets in each sample) is used to constrain the decision policy of BO to certain regions. This physical information is simulated using informed priors and the posterior means are obtained via the No-U-Turn Sampler (NUTS) extension to Hamilitonian Monte Carlo (HMC) sampling.

### [2] GPyOpt_constraints
Contains the modified GPyOpt package information. This package modifies the existing expected improvement (EI) acquisition function to include a constraint data in its decision policy.
