# Adaptive CPU Auto-Scaling Using Hybrid Neural Networks and Continual Learning in Kubernetes Orchestration


---

## 2. Introduction

### 2.1 Background

Kubernetes is the dominant container orchestration platform for managing cloud-native applications at scale. It automates deployment, scaling, and operation of containerized workloads while handling resource allocation, load balancing, and self-healing. This has made Kubernetes fundamental to CI/CD pipelines, where applications must reliably scale to meet changing demand while maintaining cost efficiency[1].

Managing CPU resources in Kubernetes has some  significant challenges. Traditional fixed resource allocation leads to over-provisioning and wasted spending, while under-provisioning causes performance degradation and SLA violations. Standard Kubernetes auto-scaling mechanisms like HPA operate reactively, adding pods only after CPU utilization exceeds its thresholds. This causes performance issues before scaling takes effect and fails to take any actions  upcoming resource spikes.

Adaptive auto-scaling uses predictive approaches to forecast future resource needs and preallocate capacity before problems arise. LSTM networks have shown strong performance in time-series CPU forecasting. However, production workloads are non-stationary, where patterns can change due to seasonal effects, application updates, and user behavior changes. Static models lose accuracy as workloads change from their training distribution, requiring frequent retraining that is operationally expensive and introduces periods of degraded performance[2] [3].

Hybrid neural networks and continual learning offer a solution to this stability-plasticity dilemma. Combining LSTM (Long Short-Term Memory)  for temporal patterns with MLPs (Multi Layer Perceptron) for static features captures both sequential dependencies and contextual information. Continual learning methods like  Elastic Weight Consolidation (EWC) and Experience Replay (ER) enable adaptation to new patterns while preserving prior knowledge, preventing catastrophic forgetting. EWC protects important parameters from updates, while ER revisits past workload samples during new training.

This research integrates hybrid neural networks with continual learning for adaptive CPU management in Kubernetes CI/CD pipelines. While continual learning has succeeded in robotics and computer vision, its application to DevOps resource management remains unexplored. This approach enables continuous adaptation to evolving workloads without forgetting historical behaviors, improving scaling accuracy, reducing over-provisioning, and enhancing system reliability.

%% Cloud native applications and containers have changed how modern software is developed and deployed. Different DevOps practices, which  mainly focus on continuous integration and continuous deployment (CI/CD), will be based  on efficient resource management to be more reliable  while keeping  the infrastructure costs under control. Container orchestration platforms, especially Kubernetes, are now the Major approach for running containerized workloads at scale across the industry[1].

Managing CPU resources in dynamic environments presents major challenges. Traditional resource allocation  planning assigns fixed resources based on peak loads of the system, which can often lead  to overprovisioning and wasted resources. Under-provisioning, on the other hand, causes slowdowns, Service Level Agreement (SLA) violations, and outages during sudden demand spikes  in the application . Achieving the right balance between cost and performance requires adaptive and  more intelligent resource allocation.

Predictive auto-scaling uses machine learning models to forecast resource needs and preallocate capacity before problems occur. Unlike reactive threshold-based methods that respond only after resource limits are exceeded, predictive approaches are based on past workload patterns to estimate future demands. LSTM (Long Short-Term Memory) networks in particular have shown strong performance in time-series forecasting of CPU utilization[2].

However, static machine learning models lose accuracy when workload patterns shift away from the training distribution. Production systems are essentially non-stationary. Seasonal effects, application updates, user behavior changes, and infrastructure evolution continually affect the  resource usage patterns. A common  solution will be to  frequently retrain the model , which is operationally expensive and introduces different periods where the model  can underperform[3].

Continual learning addresses this issue by allowing neural networks to adapt to new patterns while keeping the  prior learned knowledge. Techniques such as Elastic Weight Consolidation (EWC) and Experience Replay (ER) reduce catastrophic forgetting, where new training results may replace  earlier learning. While these methods have seen success in robotics, computer vision, and natural language processing, they remain under explored in DevOps and resource management ascpects.

This research will connect predictive auto-scaling with continual learning by introducing a framework that combines hybrid neural networks (LSTM + MLP) with continual learning techniques (EWC + ER) for adaptive CPU management in Kubernetes-based CI/CD pipelines. 
By combining these pattern recognition with static feature processing and enabling ongoing results to make changes to the  workloads, the framework is expected  to improve scaling accuracy, reduce over-provisioning, and increase the overall system reliability .

Standard Kubernetes auto-scaling, such as the Horizontal Pod Autoscaler (HPA), is mainly  based on reactive approach. It adds pods only after CPU utilization passes a predefined limit, which can lead to performance degradation before scaling actions are taken effect. Threshold-based policies may also move around boundary conditions and fail to expect upcoming spikes in resource requirement.

Predictive auto-scaling addresses these issues by forecasting future demand and preparing resources in advance to the expected usage. Deep learning models, mainly LSTM networks, are best  for CPU metric forecasting. But most existing approaches assume stable workloads and depend on offline retraining of the model.

In real DevOps environments, workloads changes continuously:

- New services and features may change the  performance profiles.
- CI/CD pipelines introduce cloud bursting, job-based resource patterns.
- User behavior, seasonality, and deployment strategies may change over time.

When models are retrained on new data, they may face the plasticity–stability dilemma:

- **Plasticity:** the ability to quickly adapt to new workloads.
- **Stability:** the ability to retain previously learned patterns.

Continual learning techniques like EWC(Elastic Weight Consolidation), which keeps the  important parameters, and ER( Elastic Research) , which revisits past samples, offer  solutions. Their application in Kubernetes autoscaling and DevOps CPU forecasting, however, is still largely not explored. %%

%% #### Proposed Framework

This research will introduce a predictive CPU auto-scaling framework for Kubernetes featuring:

- A hybrid two-branch neural network (LSTM for temporal features and MLP for static features).
- Continual learning using EWC and ER.
- Evaluation on workloads such as Alibaba Traces modifying with CI/CD patterns. %%



### 2.2 Problem Definition

Despite progress in machine learning-based auto-scaling, current systems face major limitations in real DevOps environments.

**1. Static Model Limitations**

Existing predictive systems rely on static models trained on past data. These models assume that future workloads follow historical patterns. Production environments, however, are dynamic:

- Application updates change resource profiles.
- Shifts in user behavior alter traffic patterns.
- Infrastructure updates modify resource availability.
- Mixed workloads present diverse resource needs.

Static models degrade as patterns shift and require periodic retraining, increasing operational overhead[4].

**2. Catastrophic Forgetting During Model Updates**

Retraining on new data causes models to forget earlier patterns. In auto-scaling, this leads to:

- Forgetting baseline behavior after peak-period training.
- Treating anomalies as normal due to recent data bias.
- Losing the ability to handle recurring older patterns.

This reduces the effectiveness of periodic retraining.

**3. Single-Architecture Constraints**

Many solutions use either LSTM for temporal patterns or MLP for static features, but not both. Hybrid architectures outperform single-branch models in many domains yet remain underutilized in auto-scaling.

**4. Reactive Threshold-Based Fallback**

Systems like the Kubernetes HPA respond only after CPU thresholds are exceeded:

- Scaling occurs too late, causing performance drops.
- No anticipation of spikes leads to increased latency.
- Oscillatory behavior introduces instability.
- Decisions ignore temporal context.

**5. Limited DevOps-Specific Solutions**

Most research focuses on web workloads, but CI/CD pipelines have unique characteristics:

- Bursty demands driven by builds and tests.
- Stage dependencies creating sequential resource needs.
- Distinct resource profiles for compilation, testing, and deployment.
- Strict timing constraints for fast developer feedback[5].

**6. Business–Technical Alignment Gap**

Academic work often prioritizes technical metrics over business goals such as cost tradeoffs, SLA compliance, and return on investment.

#### Core Problem Statement

Develop a CPU forecasting and auto-scaling system for Kubernetes that can adapt continuously to evolving workloads without forgetting established patterns, while lowering cost and improving reliability in DevOps environments.

This problem decomposes into:

- The limits of static models and reactive scaling.
- The plasticity–stability trade-off.
- DevOps-specific workload challenges.

A suitable solution must:

- Forecast temporal CPU demand and static job context.
- Adapt online to new workload patterns while retaining past knowledge.
- Integrate smoothly with Kubernetes-based DevOps tools.


Despite progress in machine learning-based auto-scaling, current systems face major limitations in real DevOps environments where static models trained on historical data degrade as workloads evolve through application updates, user behavior shifts, and infrastructure changes, requiring costly periodic retraining that causes  catastrophic forgetting where it causes the models to lose their ability to handle baseline or recurring patterns after learning from recent data. Most solutions has a  single-architecture designs (LSTM or MLP alone) rather than hybrid approaches, fall back to reactive threshold-based scaling that responds only after performance degradation occurs, 
The core challenge is to develop a CPU forecasting and auto-scaling system for Kubernetes that can adapt continuously to evolving workloads without forgetting established patterns, while lowering cost and improving reliability in DevOps environments.

This problem decomposes into:

- The limits of static models and reactive scaling.
- The plasticity–stability trade-off.
- DevOps-specific workload challenges.

A suitable solution must:

- Forecast temporal CPU demand and static job context.
- Adapt online to new workload patterns while retaining past knowledge.
- Integrate smoothly with Kubernetes-based DevOps tools.

### 2.3 Research Objectives

The main goal is to develop and evaluate a continual learning-based hybrid neural framework for adaptive CPU auto-scaling in Kubernetes.

Design a predictive auto-scaling system that:

- Forecasts CPU demand using a hybrid neural model.
- Uses EWC and ER to mitigate catastrophic forgetting under changing workloads.
- Improves Kubernetes scaling decisions beyond threshold-based methods.

5
### 2.4 Motivation

The motivation for this research arises from both practical needs and academic opportunities, as cloud environments face high economic costs due to frequent overprovisioning, operational instability caused by reactive scaling delays, reduced developer efficiency from slow CI/CD pipelines, and environmental concerns linked to unnecessary energy consumption. At the same time, the academic space presents several unexplored areas, including the integration of established methods such as LSTMs, continual learning, and Kubernetes autoscaling within DevOps workflows, and the extension of continual learning techniques into time series resource management where forgetting affects real operational performance. The study also evaluates a hybrid LSTM and MLP architecture to determine whether it enhances forecasting accuracy for resource management tasks, contributes reproducible datasets and evaluation protocols to address the lack of standardized benchmarks for continual learning based autoscaling, aligns technical improvements with measurable business outcomes such as cost reduction and SLA compliance, and introduces a methodological framework that includes structured data mining, knowledge retention through EWC and ER, and a multiobjective evaluation approach that considers accuracy, responsiveness, efficiency, and business impact.

---

## 3. Related Work

### 3.1 Machine Learning-Based Auto-Scaling in Cloud and Container Environments

Machine learning approaches to auto-scaling have evolved greatly  over the past decade, increasing  from simple regression models to more advanced deep learning architectures.

#### 3.1.1 Early Statistical Approaches

In the beginning  predictive auto-scaling research used time-series forecasting methods such as ARIMA (AutoRegressive Integrated Moving Average), Holt-Winters exponential smoothing, and linear regression. Wang (2021) compared Holt-Winters and LSTM for vertical CPU autoscaling in Kubernetes, finding that while Holt-Winters provided reasonable accuracy for periodic workloads, it struggled with non-stationary patterns and required manual parameter tuning. These statistical methods demonstrated the value of prediction-based scaling but doesnt have  the capacity to model complex, non-linear workload dynamics characteristic of modern cloud environments[6], [7].

#### 3.1.2 Machine Learning Era

The introduction of supervised machine learning techniques marked significant progress. Noor et al. (2019) proposed a hybrid auto-scaling system combining Random Forest for workload classification with Multiple Linear Regression for resource prediction. Their approach achieved 15-20% improvement over threshold-based methods but required extensive feature engineering and domain expertise[8].

Toka et al. (2021) developed a Kubernetes-specific scaling engine using neural networks and multiple linear regression that made application-agnostic scaling decisions handling the actual variability of incoming requests. Their work validated that ML-based approaches could generalize across diverse application types without manual tuning, addressing a key limitation of rule-based systems[9].

#### 3.1.3 Deep Learning Advancement

BHyPreC, introduced in 2021, presented a novel hybrid recurrent neural network stacking BiLSTM on top of LSTM and GRU layers for CPU workload prediction in cloud VMs. This multi-layer hybrid architecture enhanced non-linear data analysis capability, demonstrating that combining multiple RNN variants could capture complex temporal patterns better than single architectures[10].

Recent work by Singh et al. (2025) integrated LSTM-based autoscaling with Integer Linear Programming (ILP) optimization for joint autoscaling and scheduling in KubeEdge environments. Tested on an 11-node testbed, their approach used CPU, memory, and RTT metrics for prediction while optimizing container placement through ILP, addressing the coupled problem of when and where to scale resources[11].

#### 3.1.4 Production Deployment Validation

Alibaba's AHPA (Adaptive Horizontal Pod Autoscaling) system, deployed on Alibaba Cloud Container Service for Kubernetes in 2023, represents a significant industry validation. Using robust decomposition forecasting to solve the elastic lag problem, AHPA achieved 10% increase in CPU usage and over 20% reduction in resource cost compared to previous algorithms across thousands of production workloads. This large-scale deployment demonstrated that ML-based autoscaling delivers measurable business value in real-world environments[12].

The MS-RA (Microservices Requirements-driven Autoscaling) solution proposed in 2024 achieved remarkable efficiency gains: 50% less CPU time, 87% less memory, and 90% fewer replicas compared to Kubernetes HPA while meeting SLO requirements. These dramatic improvements underscore the inefficiency of reactive threshold-based approaches and validate the superiority of intelligent predictive methods[3].

### 3.2 LSTM and Recurrent Neural Networks for Workload Prediction

Long Short-Term Memory networks have emerged as the dominant architecture for time-series workload forecasting due to their ability to capture long-range temporal dependencies.

#### 3.2.1 Hybrid Recurrent Architectures

The BHyPreC model's combination of Bi-LSTM, LSTM, and GRU demonstrated that heterogeneous recurrent architectures capture complementary temporal patterns. Their research showed that:

- GRU layers efficiently capture short-term fluctuations with fewer parameters
- LSTM layers model long-term dependencies and seasonal patterns
- Bi-LSTM layers refine predictions by incorporating bidirectional context

This architectural diversity enabled better handling of workload heterogeneity the coexistence of jobs with vastly different temporal characteristics within the same cluster.

#### 3.2.2 LSTM Variants and Optimizations

Recent research has explored LSTM enhancements for resource prediction. ILP-optimized LSTM combines neural prediction with discrete optimization, demonstrating that coupling LSTM forecasts with constraint-based scheduling improves end-to-end system performance beyond prediction accuracy alone[11].

Graph-PHPA introduced LSTM-GNN (Graph Neural Networks) hybrids that model microservice dependencies explicitly. By representing microservice call graphs as graph structures and using GNN to propagate information across service boundaries, this approach captures cascading resource requirements that single-service LSTM models miss, a critical consideration for distributed applications[13].

### 3.3 Hybrid Neural Network Architectures

The combination of multiple neural network types has gone in a promising direction for capturing different types of  patterns in complex data.

#### 3.3.1 Architectural Diversity Rationale

Ghimire et al. (2022) provided in detail  analysis of CNN-LSTM-MLP hybrid fusion models for solar radiation forecasting, displaying that architectural diversity consistently outperforms single-method approaches. Their work showed:

- CNN layers excel at extracting spatial features and local patterns
- LSTM/GRU layers capture temporal dependencies and sequential dynamics
- MLP layers effectively process static features and perform non-linear transformations

Applying similar reasoning to workload prediction: LSTM captures temporal CPU usage patterns, while MLP processes static job attributes (priority, scheduling class, resource requests) that influence consumption but don't exhibit temporal variation[14].

#### 3.3.2 Hybrid Models in Resource Management

The MLP-LSTM hybrid proposed for Service Function Chain auto-scaling represents direct application to resource management. Using MLP to forecast CPU, memory, and bandwidth for Virtual Network Functions (VNFs), this architecture achieved proactive scaling that prevented performance degradation from dynamic workload variations. The explicit separation of temporal (LSTM) and static (MLP) processing paths proved computationally efficient and interpretable[15].

### 3.4 Continual Learning and Catastrophic Forgetting

Continual learning approaches address the challenge of adapting models to new data while keeping the previously acquired knowledge, critical for production systems that has  evolving workloads.

#### 3.4.1 Experience Replay Mechanisms

Rolnick et al.'s (2018) CLEAR (Continual LEArning with Replay) demonstrated that replay buffers can almost eliminate catastrophic forgetting. By maintaining a memory of representative past examples and mixing them with new data during training, Experience Replay ensures the model retains knowledge of previous patterns. The research showed that even constrained buffer sizes (storing only 1-5% of total data) provide substantial forgetting mitigation through intelligent sample selection[16].

#### 3.4.2 Continual Learning for Time-Series

CEL (Continual Learning Model for Disease Outbreak Prediction) applied EWC with LSTM for domain incremental learning on time-series data. Results showed minimal 65% forgetting rate and 18% higher memory stability compared to standard retraining. This work demonstrated EWC's effectiveness for temporal prediction tasks, though in a different application domain (epidemiology rather than resource management)[17].

#### 3.4.3 Auto-Scaling Specific Application

Hao et al.'s (2023) DMSHM (Density-based Memory Selection and Hint-based Network Learning Model) represents the only identified work explicitly addressing continual learning for predictive autoscaling[18]. Key contributions:

- Discovered sample overlap phenomenon: replay-based methods in prediction tasks exhibit unique challenges compared to classification tasks
- Introduced density-based sample selection using kernel density estimation for intelligent memory buffer construction
- Demonstrated that continual learning enables accurate predictions using only a small portion of historical logs

This work validates continual learning's applicability to auto-scaling but focuses on general cloud environments rather than DevOps-specific contexts.

### 3.5 Kubernetes Auto-Scaling and Container Orchestration

Kubernetes has become the top container orchestration platform, but its default autoscaling mechanisms shows a  significant limitations.

#### 3.5.1 Kubernetes HPA Limitations

The default Horizontal Pod Autoscaler (HPA) uses reactive threshold-based scaling approach, monitoring CPU/memory utilization and scaling when thresholds are exceeded . Research has identified critical shortcomings in this approach:

- **Reactive nature:** Scales only after resource exhaustion, causing performance degradation
- **Metric lag:** Default 15-second metric collection intervals introduce delays
- **Oscillation:** Naive threshold rules cause rapid scale-up/scale-down cycles
- **Lack of prediction:** No anticipatory capability for known patterns (daily cycles, weekly trends)

#### 3.5.2 Enhanced Kubernetes Scaling Solutions

Multiple research efforts have developed intelligent replacements for HPA:

**AHPA (Alibaba):** Deployed across Alibaba Cloud, uses robust decomposition forecasting to predict workload and proactively scale. Achieved 10% CPU usage increase and 20% cost reduction in production.[12]

**MS-RA:** Requirements-driven autoscaling achieving 50% less CPU, 87% less memory, 90% fewer replicas than HPA while meeting SLOs. Self-adaptive architecture adjusts to changing requirements without manual intervention[3].

#### 3.5.3 Kubernetes Scheduling Integration

ILP-optimized LSTM autoscaling addresses the coupled problem of scaling and placement. By jointly optimizing when to scale (via LSTM prediction) and where to place containers (via ILP), this approach achieves superior end-to-end performance compared to decoupled solutions[11].

---

## 4. Research Gaps

### 4.1 Gap 1: Lack of Continual Learning for Kubernetes Scaling

There are almost no continual learning approaches used for Kubernetes auto-scaling. Most research on Kubernetes scaling relies on:

- Threshold-based methods (e.g., HPA).
- Static ML or deep learning models trained offline.
- Reinforcement learning agents that still require periodic retraining or complex reward design.

Very few studies apply Elastic Weight Consolidation (EWC) or Experience Replay (ER) to CPU forecasting or pod auto-scaling in Kubernetes. The problem of adapting a CPU forecasting model in a live cluster without full retraining and without losing old knowledge remains largely unsolved.

### 4.2 Gap 2: Catastrophic Forgetting in DevOps is Unexplored

Catastrophic forgetting is well-known in machine learning, but it is rarely addressed in DevOps and auto-scaling research:

- When models learn new workload patterns (e.g., new pipelines or services), they forget older but still relevant patterns.
- Most auto-scaling studies focus on improving prediction accuracy or response time on current data, not on retaining performance on previous  workloads.
- Almost no work uses EWC or similar techniques in DevOps CPU forecasting to prevent catastrophic forgetting.

Using EWC and ER in this project to preserve knowledge of past workloads while learning new ones is a key new contribution.

### 4.3 Gap 3: Hybrid Temporal–Static Modeling for Auto-Scaling is Underused

Most predictive scaling solutions use either:

- Time-series models (e.g., LSTM or ARIMA) on CPU utilization alone, or a Feedforward models on static or aggregated features.

Few approaches combine both:

- Temporal sequences of CPU usage, and
- Static features (e.g., job priority, resource limits, environment metadata)

in a dedicated hybrid architecture (e.g., LSTM + MLP branches). Options for such architectures (fusion methods, branch sizes, etc.) are not well explored in resource management.

### 4.4 Gap 4: Limited Focus on DevOps/CI/CD Workload Characteristics

Most auto-scaling research targets:

- Web services with user-driven traffic, or
- General cloud workloads.

DevOps workloads, especially CI/CD pipelines run by tools like Jenkins on Kubernetes, exhibit:

- Strong job-driven behavior.
- Highly bursty usage patterns.
- Stage dependencies and critical path limits.

There is little research focusing specifically on these DevOps workloads or measuring improvements in pipeline performance and developer feedback time.

### 4.5 Gap 5: Few End-to-End Frameworks Link Technical Scaling to Business Goals

Many studies optimize technical metrics (e.g., prediction error, utilization) but do not connect them to:

- Cost savings.
- SLA compliance.
- Developer productivity.
- Long-term capacity planning.

There is a need for comprehensive frameworks that integrate advanced ML methods (hybrid models, continual learning) with business goals (e.g., cost-aware policies, SLA-aware decisions).




### Research Gaps

From past studies, we can see that:

1. **Lack of Continual Learning for Kubernetes Scaling**: Almost no continual learning approaches like EWC or Experience Replay have been applied to Kubernetes auto-scaling, leaving the problem of adapting models without full retraining and without forgetting old knowledge unsolved.
    
2. **Catastrophic Forgetting in DevOps is Unexplored**: While catastrophic forgetting is well-known in ML, it is rarely addressed in auto-scaling research, with most studies focusing on current data accuracy rather than retaining performance on previous workloads.
    
3. **Hybrid Temporal–Static Modeling is Underused**: Most solutions use either time-series models on CPU data or feedforward models on static features, with few combining both temporal sequences and static job characteristics in hybrid architectures.
    
4. **Limited Focus on DevOps/CI/CD Workloads**: Most auto-scaling research targets web services rather than DevOps workloads, which exhibit job-driven behavior, bursty patterns, and stage dependencies specific to CI/CD pipelines.
    
5. **Weak Link Between Technical and Business Goals**: Many studies optimize technical metrics like prediction error but fail to connect them to business outcomes such as cost savings, SLA compliance, or developer productivity.

---

## 5. Materials and Methods

### 5.1 Materials/Tools

#### 5.1.1 Dataset: Alibaba Cluster Traces

The Alibaba Cluster traces (2025) gives a  detailed telemetry from a large-scale containerized cloud environment. The dataset has difference  instance-level and node-level information over a period of one month, covering thousands of machines and tens of thousands of application instances.  Each instance includes CPU, memory, disk, and GPU resource requests and limits, as well as scheduling constraints, lifecycle timestamps, and application grouping.

Key characteristics include:

- **Scale and Duration:** Thousands of instances across thousands of nodes over one month, representing a production-grade workloads.
- **Workload Combination:** CPU-intensive, memory-intensive, and GPU-accelerated applications with different priorities and resource demands.
- **Metrics and Resolution:** Resource requests and limits for CPU, memory, GPU, disk, and RDMA bandwidth; lifecycle events such as creation, scheduling, and deletion timestamps; node roles.
- **Instance and Node Data:** Unique instance identifiers, application groupings, node roles (CPU node, heterogeneous GPU node), maximum instance per node limits, and static node attributes.

#### 5.1.2 Proposed Framework Architecture

The architecture consists of four components:

**Hybrid Neural Network**  
The hybrid model consists of an LSTM branch for temporal sequences and an MLP branch for static features. The LSTM branch receives input with a batch dimension, 24 time steps, and temporal features, and uses three LSTM layers with 128, 64, and 32 units with dropout to form a 16 dimensional embedding. The MLP branch processes static inputs using dense layers with 64, 32, and 16 units with ReLU activation, batch normalization, and dropout. The embeddings are concatenated and passed through a dense fusion layer and a final output neuron predicting CPU demand thirty minutes into the future. Training uses mean squared error loss with the Adam optimizer.

**Continual Learning Engine**  
The continual learning engine combines Elastic Weight Consolidation and Experience Replay. Elastic Weight Consolidation uses the Fisher information matrix to regularize changes to important parameters. Experience Replay maintains a memory buffer of earlier training samples, and minibatches contain a mixture of new and replayed data. The total loss combines mean squared error with the Elastic Weight Consolidation penalty.

**Auto Scaling Controller**  
The auto scaling controller uses predictions to compute desired pod counts based on target utilization thresholds. Scaling actions are rate limited and applied through the Kubernetes API. A fallback reactive controller is enabled when predictive estimates are unavailable.

**Monitoring and Feedback**  
This component tracks prediction error, service level agreement compliance, and cost. Drift is detected by evaluating prediction deviation against actual usage. Triggered updates invoke continual learning. A live dashboard provides visualizations.

#### 5.1.3 Baseline Methods for Comparison

The following baseline methods are used for comparison:

- Kubernetes Horizontal Pod Autoscaler
- Static LSTM only model
- Static hybrid model without continual learning
- Periodic retraining of the hybrid model

#### 5.1.4 Simulation Environment

A Kubernetes cluster using Minikube or kind with ten nodes is used. Google trace jobs are replayed according to their arrival times. Prometheus and Grafana are used to collect metrics.

#### ~~5.1.5 Software and Libraries~~

~~The implementation uses Python 3.9 or higher, TensorFlow or PyTorch, pandas, NumPy, scikit learn, visualization libraries, the Kubernetes client, and Prometheus and Grafana.~~

#### ~~5.1.6 Hardware~~

~~Training is performed using a High Performing GPU, along with a multicore CPU and atleast 32 GB of RAM. At least 500 GB of SSD storage is used for datasets and logs.~~

### 5.2 Procedures

#### 5.2.1 Data Preprocessing

Raw protocol buffer tables are parsed to extract relevant fields. Incomplete or corrupted records and extreme outliers are removed. CPU usage is normalized by machine capacity and timestamps are aligned to a unified reference.

Task instances are aggregated to their parent jobs by summing resource usage and computing job execution time. Jobs are labeled according to outcome such as completed, failed, killed, or evicted.

Time series of resource usage are resampled at five minute intervals. Minor gaps are interpolated and short jobs are padded to a minimum of twelve intervals to ensure uniform sequence length. Sliding windows, for example two hour windows with thirty minute steps, are constructed for sequence modeling.

Temporal features include raw CPU usage, first differences, rolling statistics such as mean, standard deviation, minimum, and maximum, and cyclic encodings for time of day and day of week. Static features include job priority, scheduling class, requested resources, machine capacity, platform ID, cluster ID, and historical usage summaries. Categorical attributes are encoded using one hot encoding or embeddings.

#### 5.2.2 Simulation Procedure

The simulation follows these steps:

1. Initialize the baseline controller.
2. Replay the workload.
3. Enable autoscaling under each method.
4. Collect metrics including CPU usage, predictions, scaling actions, SLA violations, and cost.
5. Evaluate performance across all approaches.

#### 5.2.3 Code Organization and Reproducibility

The codebase will be kept up to date, with separate components for each subsystem. Preprocessed data, model checkpoints, and experiment scripts will be archived, and comprehensive README documentation will be provided.

### 5.3 Data Analysis

#### 5.3.1 Evaluation Metrics

Prediction accuracy metrics such as MAE, RMSE, and MAPE will be measured along with scaling efficiency, operational quality including utilization and response lag, SLA compliance, cost, and continual learning metrics such as forgetting versus improvement.

#### 5.3.2 Statistical Validation

Multiple runs with different random seeds will be conducted. Results will be reported as means with 95 percent confidence intervals, paired t tests, and Cohen's d effect sizes.

---

## 6. Expected Outcomes

The proposed framework is expected to achieve the following:

1. **Improved Prediction Accuracy:** Surpass naive and threshold-based methods by leveraging both temporal and static information.

2. **Reduced Catastrophic Forgetting:** EWC and ER mechanisms enhance retention of earlier workload patterns compared to naive fine-tuning.

3. **Stable and Cost-Efficient Auto-Scaling:**
   - Lower CPU over-provisioning without increasing under-provisioning risk.
   - More consistent SLA compliance under dynamically changing workloads.

4. **Enhanced DevOps Pipeline Performance (where tested):**
   - Shorter queue times for builds and tests.
   - More predictable pipeline execution times during peak hours.

5. **Methodological Contributions:**
   - A reusable hybrid continual learning architecture for resource forecasting.
   - An evaluation protocol for sequential or continual workload learning in Kubernetes environments.

---

## 7. Ethical Approval

Ethical and responsible considerations for this research will mainly focus on privacy, fairness, reproducibility. All datasets such as Alibaba Traces will remain anonymized, ensuring that no PII (Personal Identifiable Information) or sensitive application data will be involved, and no re-identification attempts will be conducted. The system will also be evaluated for fairness so that resource allocation does not favor specific job types or priority classes, with suitable metrics used to observe differences across workloads. Reproducibility will be supported through clear documentation of code, model architectures, and experimental settings, and both code and preprocessing pipelines will be openly shared. 

---

## 8. Gantt Chart

| Work/Time (Months) | Nov-25 | Dec-25 | Jan-26 | Feb-26 | Mar-26 | Apr-26 | May-26 | Jun-26 |
|--------------------|--------|--------|--------|--------|--------|--------|--------|--------|
| Supervisor Selection and Title Selection | ▓▓▓▓ | | | | | | | |
| Proposal | | ▓▓▓▓ | | | | | | |
| Literature Review | | ▓▓▓▓ | ▓▓▓▓ | | | | | |
| Resource gathering | | | ▓▓▓▓ | ▓▓▓▓ | | | | |
| Tools and Techniques | | | | ▓▓▓▓ | ▓▓▓▓ | | | |
| Implementation | | | | | ▓▓▓▓ | ▓▓▓▓ | ▓▓▓▓ | |
| Documentation Writing | | | | | | ▓▓▓▓ | ▓▓▓▓ | ▓▓▓▓ |
| Publication / Conferences | | | | | | | ▓▓▓▓ | ▓▓▓▓ |
| Presentation and Manuscript Submission | | | | | | | | ▓▓▓▓ |

---

## References

[1] Devesh Srivastava et al. "Auto-Scaling of Cloud Applications Using Machine Learning". In: 2025 International Conference on Next Generation of Green Information and Emerging Technologies (GIET). Gunupur, India: IEEE, Aug. 2025, pp. 1–6. ISBN: 978-1-6654-5806-1. DOI: 10.1109/GIET65294.2025.11234879. URL: https://ieeexplore.ieee.org/document/11234879/ (visited on 12/11/2025).

[2] Mahmoud Imdoukh, Imtiaz Ahmad, and Mohammad Gh. Alfailakawi. "Machine learning-based auto-scaling for containerized applications". en. In: Neural Comput & Applic 32.13 (July 2020), pp. 9745–9760. ISSN: 0941-0643, 1433-3058. DOI: 10.1007/s00521-019-04507-z. URL: http://link.springer.com/10.1007/s00521-019-04507-z (visited on 12/11/2025).

[3] Joao Paulo Karol Santos Nunes et al. Self-adaptive, Requirements-driven Autoscaling of Microservices. arXiv:2403.08798 [cs]. Feb. 2024. DOI: 10.48550/arXiv.2403.08798. URL: http://arxiv.org/abs/2403.08798 (visited on 12/11/2025).

[4] Sukriti Srivastava. AWS Auto Scaling: Optimize Performance & Reduce Costs. Feb. 2025. URL: https://metadesignsolutions.com/aws-auto-scaling-optimize-performance-reduce-costs/ (visited on 12/12/2025).

[5] R. Premalatha et al. "Enhancing Domain Adaptation in Continual Learning with Elastic Weight Consolidation: A Multi-Dataset Deep Learning Approach". en. In: Pattern Recognition. ICPR 2024 International Workshops and Challenges. Ed. by Shivakumara Palaiahnakote et al. Vol. 15615. Series Title: Lecture Notes in Computer Science. Cham: Springer Nature Switzerland, 2025, pp. 195–209. ISBN: 978-3-031-87659-2 978-3-031-87660-8. DOI: 10.1007/978-3-031-87660-8_15. URL: https://link.springer.com/10.1007/978-3-031-87660-8_15 (visited on 12/11/2025).

[6] Abhishek Aich. Elastic Weight Consolidation (EWC): Nuts and Bolts. arXiv:2105.04093 [cs]. May 2021. DOI: 10.48550/arXiv.2105.04093. URL: http://arxiv.org/abs/2105.04093 (visited on 12/11/2025).

[7] Haibin Yuan and Shengchen Liao. "A Time Series-Based Approach to Elastic Kubernetes Scaling". en. In: Electronics 13.2 (Jan. 2024), p. 285. ISSN: 2079-9292. DOI: 10.3390/electronics13020285. URL: https://www.mdpi.com/2079-9292/13/2/285 (visited on 12/11/2025).

[8] Adnan Umer, Adnan Noor Mian, and Omer Rana. "Predicting machine behavior from Google cluster workload traces". en. In: Concurrency and Computation 35.5 (Feb. 2023), e7559. ISSN: 1532-0626, 1532-0634. DOI: 10.1002/cpe.7559. URL: https://onlinelibrary.wiley.com/doi/10.1002/cpe.7559 (visited on 12/11/2025).

[9] Laszlo Toka et al. "Machine Learning-Based Scaling Management for Kubernetes Edge Clusters". In: IEEE Trans. Netw. Serv. Manage. 18.1 (Mar. 2021), pp. 958–972. ISSN: 1932-4537, 2373-7379. DOI: 10.1109/TNSM.2021.3052837. URL: https://ieeexplore.ieee.org/document/9328525/ (visited on 12/11/2025).

[10] Md. Ebtidaul Karim et al. "BHyPreC: A Novel Bi-LSTM Based Hybrid Recurrent Neural Network Model to Predict the CPU Workload of Cloud Virtual Machine". In: IEEE Access 9 (2021), pp. 131476–131495. ISSN: 2169-3536. DOI: 10.1109/ACCESS.2021.3113714. URL: https://ieeexplore.ieee.org/document/9540844/ (visited on 12/11/2025).

[11] Shivan Singh et al. "ILP Optimized LSTM-based Autoscaling and Scheduling of Containers in Edge-cloud Environment". In: JTIT 2 (June 2025), pp. 56–68. ISSN: 1899-8852, 1509-4553. DOI: 10.26636/jtit.2025.2.2088. URL: https://www.jtit.pl/jtit/article/view/2088 (visited on 12/11/2025).

[12] Zhiqiang Zhou et al. AHPA: Adaptive Horizontal Pod Autoscaling Systems on Alibaba Cloud Container Service for Kubernetes. arXiv:2303.03640 [cs]. Mar. 2023. DOI: 10.48550/arXiv.2303.03640. URL: http://arxiv.org/abs/2303.03640 (visited on 12/11/2025).

[13] Hoa X. Nguyen, Shaoshu Zhu, and Mingming Liu. Graph-PHPA: Graph-based Proactive Horizontal Pod Autoscaling for Microservices using LSTM-GNN. arXiv:2209.02551 [cs]. Sept. 2022. DOI: 10.48550/arXiv.2209.02551. URL: http://arxiv.org/abs/2209.02551 (visited on 12/11/2025).

[14] Sujan Ghimire et al. "Deep learning CNN-LSTM-MLP hybrid fusion model for feature optimizations and daily solar radiation prediction". en. In: Measurement 202 (Oct. 2022), p. 111759. ISSN: 02632241. DOI: 10.1016/j.measurement.2022.111759. URL: https://linkinghub.elsevier.com/retrieve/pii/S0263224122009629 (visited on 12/11/2025).

[15] Sabidur Rahman et al. Auto-Scaling Network Resources using Machine Learning to Improve QoS and Reduce Cost. arXiv:1808.02975 [cs]. Mar. 2019. DOI: 10.48550/arXiv.1808.02975. URL: http://arxiv.org/abs/1808.02975 (visited on 12/11/2025).

[16] David Rolnick et al. Experience Replay for Continual Learning. arXiv:1811.11682 [cs]. Nov. 2019. DOI: 10.48550/arXiv.1811.11682. URL: http://arxiv.org/abs/1811.11682 (visited on 12/11/2025).

[17] Saba Aslam et al. "CEL: A Continual Learning Model for Disease Outbreak Prediction by Leveraging Domain Adaptation via Elastic Weight Consolidation". en. In: Interdiscip Sci Comput Life Sci 17.2 (June 2025), pp. 390–408. ISSN: 1913-2751, 1867-1462. DOI: 10.1007/s12539-024-00675-2. URL: https://link.springer.com/10.1007/s12539-024-00675-2 (visited on 12/11/2025).

[18] Hongyan Hao et al. "Continual Learning in Predictive Autoscaling". In: Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. arXiv:2307.15941 [cs]. Oct. 2023, pp. 4616–4622. DOI: 10.1145/3583780.3615463. URL: http://arxiv.org/abs/2307.15941 (visited on 12/11/2025).
