# Disaster-Decision-Support-using-Game-Theory
## Introduction

This project is an interactive decision support tool for a disaster management authority. The user describes a hypothetical flood: which regions are affected, how many people live there, how severe the flooding is reported to be, how many rescue boats, rescue teams and food packets are available, what the flood might do next, and how reliable the incoming reports are. With one click, the system analyses the situation using several game theory models and recommends how the limited resources could be allocated.

The tool runs as a web dashboard that opens in a browser tab, and it is launched from a Jupyter notebook. It contains six tabs: Configuration, Map, Game theory, Allocation, What-if, and How it was processed.

Files in this project:

- GameTheory_Disaster_DSS.ipynb : the notebook that contains the solvers, the dashboard and the launcher.
- dss_app.py : the same application as a standalone script, for running without Jupyter.
- README.md : this document.

## Why this project

In a real flood there are never enough boats, teams and supplies for everyone, and the decision maker rarely has complete information. Three difficulties appear together:

1. Resources are scarce, so every allocation is a trade-off between regions.
2. Information is uncertain. A region may report a situation that cannot be verified, and the flood itself may stabilise, worsen or become extreme.
3. The parties involved have competing interests. Regions want help for themselves, and some may exaggerate their reports.

A simple rule such as dividing supplies by population ignores all three. This project exists to show that these difficulties can be modelled explicitly and that the resulting analysis can guide a more defensible allocation.

The application is a decision support tool for exploring hypothetical scenarios. It does not predict how real communities will behave, and it is not a flood forecasting model.

## How game theory helps

Game theory is the mathematical study of decisions made by several parties whose outcomes depend on one another, often under uncertainty. A disaster response fits this description closely:

- The authority and the regions are decision makers whose choices affect each other.
- The outcome for one region depends on what the other regions claim and receive.
- Beliefs about the true situation change as new information arrives.
- Planning has to hold up even if the probability estimates turn out to be wrong.

Game theory provides a precise method for each of these. In this project the game theory models drive the calculations. The map is only the visual interface.

## Concepts of game theory used

All of the following are computed by the code, not only described.

**1. Bayesian reasoning (incomplete information)**
The authority cannot verify a region's true severity, which may be Moderate, Severe or Critical. The user enters a prior belief and a report reliability. The system applies Bayes' rule to produce a posterior probability for each severity level, and from it an expected severity for each region.

**2. Expected utility**
For each possible future (for example Stable, Worsens, Extreme), the system measures how well an allocation serves the people affected. Utility for a region is its population multiplied by its expected severity multiplied by the average share of its boat, team and food needs that are covered, with coverage capped at 100 percent. The probability-weighted average across futures gives the expected utility of an allocation. The system searches a grid of possible allocations to find the best one.

**3. Minimax (worst-case decision making)**
If the probability estimates cannot be trusted, the authority can instead choose the allocation whose worst outcome across all futures is as good as possible. The system computes this allocation and compares it with the expected-utility choice.

**4. Non-cooperative game and Nash equilibrium**
Each region chooses between cooperating (claiming a fair, need-based share) and acting selfishly (claiming more). The payoffs come from the allocation model, with a conflict cost when selfish regions clash. The system checks every combination of choices and marks the pure-strategy Nash equilibria, meaning the combinations where no region gains by changing its choice alone. This is an analysis of the mathematical game that was defined, not a prediction of real behaviour.

**5. Cooperative game and Shapley value**
If regions pool their resources, the system computes the value each possible coalition could achieve, and then the Shapley value of each region, which distributes the total benefit according to each region's average marginal contribution across all orderings of joining. Each region's own endowment can be entered, otherwise it is assumed proportional to population.

**6. Stackelberg game (leader and followers)**
The authority moves first and commits to how strongly the allocation depends on reported severity versus population. The regions then respond by reporting honestly or inflating their reports, facing a chance of audit and a penalty if caught. The authority anticipates these responses and selects the commitment that gives the best overall result.

**7. Repeated game**
The system simulates several rounds with changing flood conditions. It compares three paths: everyone cooperating, everyone acting selfishly, and one region defecting in round two followed by grim-trigger punishment. It reports cumulative welfare and the defecting region's own payoff, showing whether cooperation is worth sustaining.

**How the recommendation is formed**
The final allocation blends three components: the expected-utility optimum, the minimax optimum and the Shapley shares.

final share = (1 - fairness) x [ (1 - risk aversion) x expected-utility share + risk aversion x minimax share ] + fairness x Shapley share

The risk aversion and fairness weights are set by the user. This blend is a documented design choice, not a theorem. The Nash, Stackelberg and repeated-game results are shown as strategic stability diagnostics and do not change the blend.

**Not implemented**
Mechanism design is not implemented. The map heat layer is an illustrative spread of expected impact around each point and is not a flood or terrain model. The model has no accessibility factor.

## Inputs the user provides

- Regions: name, population, reported severity (Moderate, Severe, Critical), optional latitude and longitude, optional own share of resources.
- Resources: boats, rescue teams, food packets.
- Capacities: people served per boat, people served per team, packets needed per person.
- Possible futures: a name, a probability and a demand multiplier for each.
- Uncertainty: prior beliefs, severity weights and report reliability.
- Strategy settings: risk aversion, fairness weight, greed, conflict cost, audit probability, report inflation factor, loss if caught, and the sequence of futures for the repeated game.

If latitude and longitude are left blank, regions are placed automatically on land near Mumbai and Raigad. The "Load example" button fills the form with a sample flood in three regions.

## Steps of execution

1. Install Python 3.9 or newer.
2. Install the required libraries:
   pip install dash plotly pandas numpy
3. Open GameTheory_Disaster_DSS.ipynb in Jupyter Notebook, JupyterLab or VS Code.
4. Run the cells from top to bottom. The first code cell installs the libraries if needed, the second defines the solvers, the third defines the dashboard, and the last starts the server.
5. The last cell opens the dashboard in a new browser tab. If it does not open, use the link printed in the cell output, normally http://127.0.0.1:8050.
6. On the Configuration tab, fill in the scenario, or press "Load example".
7. Press "Analyse Disaster Scenario". The page jumps to the Allocation tab.
8. Review the results:
   - Map: affected regions, the impact heat layer and each region's share of resources. Hover over a region for its details.
   - Game theory: the Bayesian, Nash, Shapley, Stackelberg, repeated-game and minimax panels.
   - Allocation: the recommended number of boats, teams and food packets for each region.
   - What-if: change boats, teams, probabilities or reliability and compare the new allocation with the previous one.
   - How it was processed: a step-by-step trace showing how the inputs became the recommendation.
9. To stop the server, run SERVER.shutdown() in a notebook cell.

Running without Jupyter: save dss_app.py, run "python dss_app.py", and the dashboard opens in a browser tab.

Note: if the notebook runs on a remote machine (Google Colab, a cloud notebook, JupyterHub), the address 127.0.0.1 in your browser will not reach it. On Colab the last cell provides a tunnel link. For other remote setups, run dss_app.py on your own computer instead.

## Conclusion

This project shows that disaster resource allocation can be treated as a strategic problem rather than a simple division of supplies. Bayesian updating handles unreliable reports, expected utility and minimax handle uncertainty about the future, the cooperative game gives a fair way to credit each region's contribution, and the Nash, Stackelberg and repeated-game models test whether the plan remains sensible when regions respond to it. The user supplies every assumption, and the dashboard shows how each assumption shaped the result.

The tool is meant for exploring and comparing scenarios and for supporting discussion, not for replacing the judgement of trained disaster management professionals. Its assumptions, including the utility model and the simplified map heat layer, are stated openly so that they can be questioned and improved.

Made with a thoughtful usecase
