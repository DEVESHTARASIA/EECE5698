# Q1 Results compilation
## Plots
### Policy Iteration

Base Scenario:
![PI-Base-Scenario-Values](PI-Value-a.png "Policy Iteration Base Scenario Values")
![PI-Base-Scenario-Policies](PI-Policy-a.png "Policy Iteration Base Scenario Policies")

Large Stochasticity Scenario:
![PI-Large-Stochasticity-Scenario-Values](PI-Value-b.png "Policy Iteration Large Stochasticity Values")
![PI-Large-Stochasticity-Policies](PI-Policy-b.png "Policy Iteration Large Stochasticity Policies")

Small Discount Factor Scenario:
![PI-Small-Discount-Scenario-Values](PI-Value-c.png "Policy Iteration Small Discount Factor Scenario Values")
![PI-Large-Stochasticity-Policies](PI-Policy-c.png "Policy Iteration Small Discount Factor Scenario Policies")

### Value Iteration

Base Scenario:
![VI-Base-Scenario-Values](VI-Value-a.png "Value Iteration Base Scenario Values")
![VI-Base-Scenario-Policies](VI-Policy-a.png "Value Iteration Base Scenario Policies")

Large Stochasticity Scenario:
![VI-Large-Stochasticity-Scenario-Values](VI-Value-b.png "Value Iteration Large Stochasticity Values")
![VI-Large-Stochasticity-Policies](VI-Policy-b.png "Value Iteration Large Stochasticity Policies")

Small Discount Factor Scenario:
![VI-Small-Discount-Scenario-Values](VI-Value-c.png "Value Iteration Small Discount Factor Scenario Values")
![VI-Large-Stochasticity-Policies](VI-Policy-c.png "Value Iteration Small Discount Factor Scenario Policies")


## Observations:
In policy iteration method, in scenarios with larger gamma value, the policy from start position to goal position came out as correct. The paths were different. 

But in the case of small discount factor, since the gamma is less which means it is less reliant on future awards. Thus 'correct' policy stays only near few of cells near the goal state or state with highest reward. The policy from start state does not terminate in the goal state. The change in values decreases less and as the states go away from the goal state.

The path in large base case scenario is longer compared to larger stochastic scenario. In general it is seen that the accumulated reward at in large stochastic case is less compared to base case scenario. In large stochastic case the more oil states are hit, in base case more bump states are hit.

Same results were observed for value iteration method. 
