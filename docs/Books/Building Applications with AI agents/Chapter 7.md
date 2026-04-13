# Chapter 7. Learning in Agentic Systems 

Adding the capability to learn and improve over time is an useful addition but not necessary while designing agents. By learning we mean improving the performance of agentic system through interaction with the environment. 

Non parametric learning refers to techniques to change and improve performance automatically without changing the models weights. Parametric learning refers to techniques in which we specifically train or finetune the parameters of the foundation model. 


## Nonparametric Learning 

### Nonparametric Exemplar Learning 

As the agent performs its task it is provided with a measure of quality and these examples are used to improve its performance. These examples are used as few-shot examples for in-context learning. If we have more examples performance improvement eventually comes with an additional cost and latency. 

Not all examples might be useful for all inputs. Common way to address this is to dynamically select the most relevant examples and add them to the context. This typically involves building a memory bank where details of each interaction like context, actions, outcomes or any feedback received are stored. This database acts more like a human memory where past experiences shape the understanding and guide future actions. 

The agent uses past cases to solve new problems. Each case includes a problem, the solution used, and the result. When a new situation arises, the agent finds similar past cases and adapts their solutions to fit the current problem. As the number of successful examples grows, it becomes useful to retrieve the most relevant ones using type-based, text-based, or semantic retrieval methods.


### Reflexion 

Reflexion helps an agent learn from mistakes by having it briefly reflect after a failed attempt. The agent notes what went wrong and how to improve next time. These reflections are stored in memory and reviewed before the next attempt, allowing the agent to adjust its strategy without retraining the model.

#### Reflexion Process

- The agent performs a task using its normal planning and actions.
- The actions, observations, and outcome are logged.
- If the task fails, the agent generates a short reflection on what to change next time.
- This reflection is saved to memory.
- On the next attempt, the agent includes the reflection in its prompt to guide better decisions.



### Experiential Learning

Experiential learning extends nonparametric learning by not only storing past experiences but also extracting insights from them to improve future behavior. The agent reflects on successes and failures, develops new techniques, and updates its approach over time.

These insights are stored and continuously refined. Useful insights are promoted, less helpful ones are downvoted or removed, and existing insights are revised as new experiences are gained.

This approach builds on Reflexion by enabling cross-task learning. It allows the agent to transfer useful strategies across different tasks. In Experiential Learning (ExpeL), the agent maintains a list of insights derived from past experiences. Over time, this list evolves as insights are added, edited, upvoted, downvoted, or removed.

With enough feedback, this approach allows an agent to learn efficiently from its interactions and improve over time. It also helps the agent adapt gradually to changing (nonstationary) environments by updating its behavior as conditions evolve.

These methods are practical, low-cost, and easy to implement, making them well suited for continual learning. In some cases, however—especially when large amounts of data are available—it may be beneficial to consider fine-tuning instead.


## Parametric Learning 

Parametric learning involves adjusting the parameters of a predefined model to improve its performance on specific tasks. It often makes sense to start with nonparametric approaches, because they are simpler and faster to implement. When we have a sufficient number of examples, it might be worth considering fine-tuning your models as well to improve your agentic performance on your tasks


### Fine-Tuning Large Foundation Models

