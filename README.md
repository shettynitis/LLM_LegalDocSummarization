## Legal Document Summarizer

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
| All team members  | - Overall project idea<br>- High-level system design<br>- CI/CD and Continuous Training (Unit 3) <br>- Final integration & docs | [All contributions](https://github.com/shettynitis/LLM_LegalDocSummarization/commits)                         |
| Navya Kriti     | - Model training (Units 4 & 5)<br>- LoRA fine-tuning with Ray cluster<br>- Experiment tracking & logging (MLflow)            | [Commits by Navya](https://github.com/shettynitis/LLM_LegalDocSummarization/commits?author=Navya0203)   |
| Nitisha Shetty    | - Model serving & monitoring (Units 6 & 7)<br>- Building inference API<br>- Load testing, staged & canary deployment         | [Commits by Nitisha](https://github.com/shettynitis/LLM_LegalDocSummarization/commits?author=shettynitis) |
| Sakshi Goenka       | - Data pipeline (Unit 8)<br>- Persistent storage & data ingestion<br>- Online/offline data management                        | [Commits by Sakshi](https://github.com/shettynitis/LLM_LegalDocSummarization/commits?author=robo-ro)     |


### System Diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->
![image](https://github.com/user-attachments/assets/0f673b10-9849-4c06-9273-4a2115a3389d)

### Summary of Outside Materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

| Name / Version                | How It Was Created                                                                                                  | Conditions of Use                                                               |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Zenodo Legal Dataset**      | Curated from publicly available court rulings, case files, and legal judgments. | |
| **Additional Legal Docs**     | Internal or publicly released legal documents from open data portals; combined and pre-processed by the team.       |  |
| **Llama-2-7b** (Meta)        | Pretrained by Meta on diverse text sources. Available via Hugging Face with special access.                          | Must adhere to Meta’s Model License and Hugging Face usage agreements.           |
| **FAISS** (Vector Store)      | Open-source library for efficient similarity search. Developed by Facebook AI Research.                             | Apache-2.0 license; can be used and distributed freely.                          |
| **LoRA / `trl` Library**      | Open-source Python library enabling parameter-efficient fine-tuning of LLMs.                                        | Various open-source licenses.     |



### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->
| Requirement               | How many/when                                 | Justification                                                                                                                        |
|---------------------------|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **`m1.medium` VMs**       | 2 permanent VMs + 1 staging VM               | - **2 permanent**: one hosts the production RAG API and FAISS index, the other runs MLflow and DevOps automation<br>- **1 staging**: used for load testing, integration tests, and canary deployment before going live                                                                                                                     |
| **GPU nodes (`A100` or `mi100`)** | 2 nodes for ~4 hours/session, up to 2x/week | - Needed for fine-tuning Llama-2-7b with LoRA<br>- Each session covers hyperparameter tuning or retraining on new data<br>- If doing distributed training, 2 GPUs can be used concurrently for faster experimentation                                                                                                                 |
| **Floating IPs**          | 1 permanent for production + 1 on-demand for staging | - **Production**: Exposes the API externally so users (or graders) can query the summarization service<br>- **On-demand**: Temporarily attach to staging when we need external testing or demos                                                                                                                                    |
| **Persistent Volume (~50–100GB)** | Attached for the entire project       | - Stores the legal dataset (Zenodo files), preprocessed text chunks, fine-tuned model artifacts, and any large logs<br>- Avoids having to re-download or re-process data every time we redeploy                                                                                            |


### Detailed Design Plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

1. Strategy
* We will fine-tune Llama-2-7b using LoRA on a Ray cluster (GPU nodes on Chameleon).
* This ensures we can handle large-scale data, while the LoRA approach keeps GPU usage manageable.
* We will track metrics (loss, ROUGE, hyperparams) in MLflow, also hosted on Chameleon.
2. Diagram References
* In the “Training” section of our system diagram, data from the Data Pipeline feeds into the Ray cluster.
* MLflow logs are stored on a CPU node, giving us experiment histories and artifact versioning.
3. Justification
* LoRA is a parameter-efficient method for large LLM fine-tuning, reducing GPU memory overhead.
* Using Ray + MLflow ensures we can do robust scheduling and logging, consistent with class best practices.
4. Relation to Lecture Material
* Unit 4: We meet “train and re-train” by repeatedly fine-tuning Llama-2-7b on new or updated data.
* Unit 5: Self-hosted MLflow on Chameleon tracks all runs. A Ray cluster schedules training jobs (just like in the labs).
5. Specific Numbers
* GPU usage: ~3–6 GPU hours per training session (depending on dataset size).
* Data: ~10,000 legal documents from Zenodo, chunked to ~512–1,024 tokens each.
* Extra Difficulty (Unit 4/5):
  - Using Ray Train
  - Training strategies for large models


#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

1. Strategy
* We will containerize a FastAPI-based inference service that does Retrieval-Augmented Generation (RAG).
* At inference, a FAISS index (or TF-IDF) retrieves relevant docs; the fine-tuned Llama-2-7b model then summarizes them.
* We’ll do load testing in staging, then optionally a canary environment before production.
* Online monitoring will capture performance metrics and user feedback.
2. Diagram References
* The “Deployment” section: RAG Vector DB + Summarization Model containers.
* The “Monitoring” block receives logs and metrics from production.
3. Justification
* Splitting out retrieval from generation is standard RAG practice: it reduces total model memory usage and ensures relevant documents are provided.
* We will track concurrency (aiming for ~5–7 concurrent requests) and keep latencies as low as possible.
4. Relation to Lecture Material
* Unit 6: An API endpoint with explicit performance/latency goals. Possibly use 4-bit or 8-bit quantization to optimize.
* Unit 7: We incorporate offline evaluation (ROUGE), load testing in staging, canary deployment, and continuous monitoring.
5. Specific Numbers
* Latency: <10 seconds on average for one summarization query.
* Load: ~20 RPS in staging to test concurrency.
* Close the Loop: If user feedback indicates poor results, we mark those examples for future re-training.

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

1. Strategy
* Persistent storage on Chameleon (50–100 GB) holds the chunked legal dataset, model artifacts, etc.
* Python ETL scripts fetch, clean, and split raw legal texts from Zenodo or other sources.
* The pipeline is triggered whenever new data arrives.
2. Diagram References
* The “Data Source” box in the system design feeds training data into the Ray cluster.
* We store outputs in the “Persistent Storage” volume, accessible by both training and serving components.
3. Justification
* Offline data management ensures stable re-training.
* If the system sees newly labeled or user-provided data, we can incorporate it into the next training cycle.
4. Relation to Lecture Material
* Unit 8: Following Lab 8’s approach to storage provisioning(as per the project manual), we will attach a volume or use object storage on Chameleon.
* We also simulate real-time data by generating user queries to test the pipeline in staging/canary.
5. Specific Numbers
* 50–100 GB capacity volume.
* Data updated weekly or on a manual schedule.


#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->

1.Strategy
* We define infrastructure-as-code using Terraform or python-chi to spin up CPU/GPU nodes on Chameleon.
* Our CI/CD pipeline automates:
- Retraining on new data,
- Running offline eval,
- Packaging the model in a Docker image,
- Deploying to staging for tests,
- Promoting to canary,
- Deploying to production.
2. Diagram References
* The central CI/CD block orchestrates the entire pipeline.
* Staging, canary, and production are distinct nodes or container groups in the our deplyment block.
4. Justification
* Minimizes “clickOps,” ensures reproducibility and version control.
* Cloud-native approach: everything is containerized, and changes are made by updating Git, not manually configuring servers.
5. Relation to Lecture Material
* Unit 3: Microservices in containers, staged deployments, and an automated pipeline for continuous training & integration.
* We follow lab patterns: no manual edits to running instances (immutable infra).
6. Specific Numbers
* 1–2 GPU nodes for training, 2–3 CPU nodes for staging, production, MLflow, etc.
* Staging tests include ~20 concurrency requests, canary ~10% traffic, then full production.

