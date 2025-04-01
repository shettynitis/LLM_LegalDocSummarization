## Legal Document Summarization

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be 
used in an existing business or service. (You should not propose a system in which 
a new business or service would be developed around the machine learning system.) 
Describe the value proposition for the machine learning system. What’s the (non-ML) 
status quo used in the business or service? What business metric are you going to be 
judged on? (Note that the “service” does not have to be for general users; you can 
propose a system for a science problem, for example.)
-->

### Value Proposition

Law firms and legal departments invest significant time and cost into manually reading, annotating, and summarizing lengthy legal documents—judgments, case files, contracts. Our system addresses this gap by fine-tuning Llama-2-7b with a Retrieval-Augmented Generation (RAG) pipeline to:

1. Identify relevant documents from a repository
2. Automatically generate accurate, concise summaries

Our project is similar to existing ML-based legal software like Kira Systems and Casetext that tackles contract review and search.

**Business Metrics**
* Time Saved: Fewer hours spent summarizing.
* Cost Reduction: Reallocate billable hours to higher-value tasks.
* Quality: Measure summary completeness (e.g., ROUGE scores, user feedback).
  
By integrating automated summarization into existing legal workflows, this system aims to streamline document handling while maintaining high-quality analysis.


### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions in this repo (or multiple repos if you split your code). -->

| Name              | Responsible for                                                                                                              | Link to their commits in this repo                                                        |
|-------------------|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| All team members  | - Overall project idea<br>- High-level system design<br>- CI/CD and Continuous Training (Unit 3) <br>- Final integration & docs | [All contributions](https://github.com/shettynitis/LLM_LegalDocSummarization/edit/main/README.md)                         |
| Navya Kriti     | - Model training (Units 4 & 5)<br>- LoRA fine-tuning with Ray cluster<br>- Experiment tracking & logging (MLflow)            | [Commits by Navya](https://github.com/shettynitis/LLM_LegalDocSummarization/edit/main/README.md)   |
| Nitisha Shetty    | - Model serving & monitoring (Units 6 & 7)<br>- Building inference API<br>- Load testing, staged & canary deployment         | [Commits by Nitisha](https://github.com/shettynitis/LLM_LegalDocSummarization/edit/main/README.md) |
| Sakshi Goenka       | - Data pipeline (Unit 8)<br>- Persistent storage & data ingestion<br>- Online/offline data management                        | [Commits by Sakshi](https://github.com/shettynitis/LLM_LegalDocSummarization/edit/main/README.md)     |


### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| Data set 1   |                    |                   |
| Data set 2   |                    |                   |
| Base model 1 |                    |                   |
| etc          |                    |                   |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->
