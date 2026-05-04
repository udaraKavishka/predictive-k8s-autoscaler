# Rationale for Time-Series Dataset Selection and Design

## Why Time-Series Data Was Chosen

The research focuses on adaptive CPU auto-scaling in Kubernetes using hybrid neural networks (LSTM + MLP) and continual learning. Accurate forecasting of future CPU demand is essential for proactive scaling decisions. The Alibaba DLRM trace, while rich in event-based records, does not natively provide the regular, fixed-interval sequences required for time-series modeling and continual learning evaluation.

### Criteria for Dataset Selection
- **Relevance to Real-World Workloads:** Alibaba traces represent large-scale, production-grade containerized environments, closely matching the target use case.
- **Granularity and Coverage:** The dataset includes detailed instance-level resource usage, scheduling, and lifecycle events across thousands of applications and nodes.
- **Support for Temporal and Static Features:** The data contains both time-varying (CPU, memory, GPU usage) and static/contextual (job priority, node role) attributes, enabling hybrid model design.

### Why Time-Series Transformation is Needed
- **Model Requirements:** LSTM-based models require regularly spaced, ordered sequences for effective learning. The hybrid model also needs static features aligned with each time step.
- **Kubernetes Scaling Needs:** Proactive scaling requires forecasts at regular intervals (e.g., every 5 minutes) to anticipate demand spikes and avoid SLA violations.
- **Continual Learning Evaluation:** EWC and Experience Replay methods depend on stable temporal segmentation (chunks) to measure adaptation and forgetting.

## Dataset Transformation Process
- **Source:** Alibaba/disaggregated_DLRM_trace.csv (event-based instance records)
- **Transformation:** Aggregated into 5-minute bins per application, producing a time-series with columns for CPU demand, memory/GPU/RDMA profiles, instance counts, and static features.
- **Output:** alibaba_timeseries_full.csv, used as the main input for all model training and evaluation.

## Model and Current Status
- **Model Creation:** Implemented the hybrid LSTM+MLP model and baseline models as planned.
- **Dataset Preparation:** Completed the transformation of Alibaba trace data into the required time-series format, ensuring compatibility with the model architecture and research objectives.
- **Current Results:** The proposed architecture is not yet performing as expected, with some metrics (e.g., MAPE, BWT) indicating room for improvement.
- **Next Steps:** Preparing a Kubernetes node for real-world testing and further validation of the model's effectiveness in a live environment.

---

This approach ensures the research remains aligned with both the technical requirements of the models and the practical needs of Kubernetes-based auto-scaling.
