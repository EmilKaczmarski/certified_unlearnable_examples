# **Project Handover: PUE Defense in Federated Learning**

## **1. Project Overview**

  * **Thesis Title (Working):** *Title to be determined*
  * **Author:** Emil Kaczmarski (s1124172)
  * **Supervisor:** Stjepan Picek, Radboud University
  * **Core Research Topic:** Investigating the effectiveness of **Provably Unlearnable Examples (PUEs)** as a data-centric defense against **Model Distillation (MD)** attacks within a **Federated Learning (FL)** environment.
  * **Primary Goal:** To analyze the trade-off between the defensive capabilities of PUEs (measured by the degradation of an adversary's stolen model) and the impact on the utility (i.e., accuracy, convergence) of the legitimate, collaboratively trained FL model.

-----

## **2. Current Status & Next Steps**

### **Writing Status**

  * The main LaTeX document structure for the thesis is established.
  * The **Introduction** section is partially complete. The following subsections have been drafted:
      * `1.1 Background and Motivation`
      * `1.2 Problem Statement`
      * `1.3 Research Questions`
      * `1.4 Thesis Outline`
  * **Next Writing Task:** Draft the `\subsection{Contributions}`.

### **Experimental Status & Key Findings**

  * **Core Discrepancy Identified:** A critical difference was found between the public `recover.py` script and the results published in the PUE paper.
      * The **paper's charts** show a **"recovery from scratch"** experiment, where a new random model is trained (low initial accuracy).
      * The **public script** performs **"model unlearning/fine-tuning,"** where it loads a pre-trained model and continues to train it (high initial accuracy).
  * **Decision Made:** To align with the paper, the plan is to **modify the `recover.py` script** to perform the "recovery from scratch" experiment.
  * **Attack Strength:** We've distinguished between the passive (`min-min`) and active (`min-max`) PUE generation methods and established that `min-max` is the stronger, more robust approach.
  * **Next Action Item:** Your new goal is to adapt this experimental framework to run with a **different model architecture**.

-----


## **3. Thesis Writing & Style Guide**

The established writing style for the thesis should be maintained for consistency.

  * **Tone:** Formal, academic, objective, and precise.
  * **Structure:** Sections and subsections should have clear, descriptive titles. Arguments should flow logically from a broad context to a specific focus. Use lists (`itemize`, `enumerate`) to organize information where appropriate.
  * **Language:** Concise and direct. Avoid conversational fillers. Key technical terms should be **bolded** upon first use or for emphasis.
  * **Citations:** Use the `\cite{...}` command for all academic sources, ensuring they are correctly listed in your `.bib` file.

-----

## **4. Key Methodological & Technical Details**

### **Models**

  * **Current Architecture:** `ResNet-18` is used for the FL clients' local models, the aggregated global model (the "teacher"), and the adversary's student model.
  * **New Architecture:** You will be replacing `ResNet-18` with your new choice (e.g., Vision Transformer, MobileNet, etc.). This change will need to be reflected in the configuration files (`resnet18.yaml`) and any hard-coded model initializations.

### **PUE Poisoning Workflow (`make_pue.py`)**

  * This script generates the unlearnable noise for each client's dataset.
  * **Attack Types:**
      * `--attack_type min-min`: A passive attack, assuming a standard training procedure by the victim.
      * `--attack_type min-max`: An active attack that crafts a more robust poison, assuming a defensive or adaptive victim. This is the stronger method.
  * **Bug Notice:** The `make_pue.py` script has a `TypeError` when using `min-max`. It incorrectly passes an extra `data_loader` argument to the `train()` function on line 511. This must be fixed for the active attack to run.
  * **Surrogate Model:** For the `min-max` attack, the script is designed to first train a surrogate model to high accuracy before using it to generate the poison. This is the correct methodology.

### **Defense/Recovery Workflow (`recover.py`)**

  * **Required Modification:** To replicate the paper's charts, this script must be modified to perform "recovery from scratch."
      * Load the trained/poisoned model into a separate `poison_model` variable.
      * Ensure the main `model` variable (the one that gets trained) remains randomly initialized.
  * **Recovery Data:**
      * The paper specifies using **20% of the training data** for recovery.
      * For your setup (10 clients, 5000 images each), this means using **1,000 samples per client**.
      * This is achieved by setting `--recover_rate 0.2` and using the `--use_train_subset` flag.

### **Key Metrics**

  * **Clean Accuracy:** Measures the utility of the FL model on a standard, clean test set.
  * **Attack Success Rate (ASR):** Measures the effectiveness of the MD attack. It is the test accuracy of the adversary's student model. A low ASR indicates a successful defense.

-----

## **5. Completed Thesis Sections (LaTeX)**

Here is the full LaTeX code for the completed introduction subsections for easy porting.

```latex
\subsection{Background and Motivation}

The proliferation of machine learning has transformed countless industries, creating powerful models that are increasingly reliant on vast quantities of data. This data is often sensitive, encompassing personal, financial, or proprietary information, which creates a fundamental conflict between the need for data to train effective models and the critical importance of preserving privacy and protecting intellectual property. Traditional centralized machine learning approaches, which require aggregating data on a single server, present significant privacy risks, making them unsuitable for many real-world applications involving sensitive user information.

Federated Learning (FL) was introduced as a paradigm-shifting solution to this problem \cite{mcmahan2017communication}. By enabling collaborative model training on decentralized data, FL allows clients to contribute to a global model without ever exposing their raw, private data. This approach has been a major catalyst for privacy-preserving machine learning, opening the door for applications in healthcare, finance, and mobile computing where data localization is a strict requirement. The core motivation behind FL is to reap the benefits of large-scale data while mitigating the systemic risks of data centralization.

However, the FL framework introduces its own unique set of vulnerabilities. While raw data remains local, the model updates (gradients or weights) that are shared during the training process can inadvertently leak sensitive information. Research has demonstrated that adversaries, particularly a malicious server, can exploit these updates to infer private information about a client's training data \cite{zhu2019deep} or even determine if a specific user's data was part of the training set \cite{shokri2017membership}. This leakage creates a new attack surface that undermines the privacy promises of FL.

Among the most significant threats is the Model Distillation (MD) attack, a form of model stealing where an adversary can create a high-fidelity copy of the valuable, collaboratively trained global model \cite{hinton2015distilling, tramer2016stealing}. Such an attack not only constitutes intellectual property theft but also enables the adversary to conduct further privacy attacks on the stolen model offline. This motivates the need for defenses that go beyond securing the communication channel and instead focus on protecting the data itself. A promising direction in this area is the concept of making data inherently "unlearnable" to unauthorized parties by adding carefully crafted perturbations \cite{huang2021unlearnable}.

While early data-centric defenses have shown empirical success, the field is increasingly moving towards solutions with formal security guarantees. This motivates the investigation into the latest generation of these techniques, namely Provably Unlearnable Examples (PUEs) \cite{yue2024provably}. PUEs offer a theoretical certificate of unlearnability, providing a robust defense that is not dependent on the adversary's specific model architecture or training procedure. The potential to integrate such a provable defense into the complex, iterative environment of Federated Learning to specifically counter Model Distillation attacks provides the central motivation for this thesis.

\subsection{Problem Statement}

Federated Learning (FL) is designed to protect raw data, but it inadvertently creates a new vulnerability: the model itself. The aggregated global model, or the updates used to create it, can be exploited by adversaries. A critical threat in this context is the **Model Distillation (MD) attack**, where an adversary with access to the final trained model can steal its intellectual property and functionality by training a surrogate student model \cite{hinton2015distilling}. This compromises the significant collaborative effort invested in the FL model and creates a privacy loophole, as the stolen model can be subjected to further offline attacks to infer information about the clients' private data.

While standard Privacy Enhancing Techniques (PETs) like Secure Aggregation \cite{bonawitz2017practical} and Differential Privacy \cite{abadi2016deep} provide protection during the training process, they primarily focus on securing the model updates in transit or providing privacy for individual data points in the final aggregate. They do not, however, inherently protect the final model from being reverse-engineered or stolen once it is deployed or if its parameters are otherwise obtained by an adversary. This leaves a significant security gap that requires data-centric defenses capable of proactively protecting the model's knowledge from being extracted.

Provably Unlearnable Examples (PUEs) have recently been proposed as a powerful, data-centric defense that provides formal, certified guarantees of unlearnability against a defined class of adversaries \cite{yue2024provably}. However, the effectiveness of PUEs has not been systematically investigated within the specific, iterative, and distributed context of Federated Learning. It is unknown whether the unlearnability properties of PUEs persist through the aggregation process and if they can effectively degrade the performance of a sophisticated MD attack. Furthermore, the impact of training an entire FL system on PUE-protected data on the final global model's utility and convergence speed is an unexplored but critical trade-off.

Therefore, the central problem this thesis addresses is the **unexplored efficacy of Provably Unlearnable Examples as a defense against Model Distillation attacks in Federated Learning environments.** Specifically, there is a critical lack of understanding regarding the trade-off between the certified security PUEs can offer against model stealing and the impact on the utility and performance of the legitimate, collaboratively trained model.

\subsection{Research Questions}

To address the problem statement, this thesis seeks to answer a central, overarching research question that encompasses the core challenges of integrating a novel defense into a complex learning framework.

\textbf{Main Research Question:}
\textit{To what extent can Provably Unlearnable Examples be integrated into a Federated Learning framework to provide a robust defense against Model Distillation attacks while maintaining the utility of the global model?}

To systematically investigate this, the research is guided by the following specific questions:

\subsubsection{Effectiveness Against Model Distillation}
How effective are PUEs in mitigating Model Distillation attacks within a Federated Learning environment? This question focuses on the defensive capabilities of the PUEs. It aims to quantify the degree to which training the FL model on PUE-protected data degrades the performance of a surrogate model trained by an adversary using Model Distillation. Success will be measured by the reduction in the stolen model's accuracy on a clean test set.

\subsubsection{Impact on Federated Learning Utility}
What is the impact of training on PUE-protected data on the utility of the legitimate global FL model? This question addresses the critical trade-off of the defense. It seeks to measure the cost of applying PUEs by evaluating the final test accuracy and convergence behavior of the global FL model compared to a baseline model trained on original, clean data.

\subsubsection{Analysis of the Security-Utility Trade-off}
What is the relationship between the certified unlearnability parameters of PUEs and the resulting trade-off between defense effectiveness and model utility? This question explores the dynamics of the defense mechanism itself. It aims to analyze how adjusting the theoretical parameters of PUEs, such as the learnability bound, influences the balance between thwarting the MD attack and preserving the legitimate model's performance.

\subsection{Thesis Outline}

The remainder of this thesis is structured as follows to systematically address the research questions:

\begin{itemize}
    \item \textbf{Chapter 2: Literature Review} provides the necessary theoretical background. It covers the foundational principles of Federated Learning, details the mechanics and implications of Model Distillation attacks, and reviews the evolution of data-centric defenses from empirical Unlearnable Examples (UEs) to the formally-grounded Provably Unlearnable Examples (PUEs). The chapter culminates in a synthesis of the literature that identifies the specific research gap this thesis aims to fill.
    \item \textbf{Chapter 3: Methodology} details the conceptual framework designed for this research. It describes the proposed methodology for integrating PUEs into a Federated Learning workflow as a defense, outlines the simulation of the Model Distillation adversary, and defines the strategies for measuring defense effectiveness and utility trade-offs.
    \item \textbf{Chapter 4: Experimental Setup} outlines the concrete implementation details of the methodology. This includes the selection of datasets (CIFAR-10 and CIFAR-100), model architectures (ResNet-18), and the FL simulation framework. It specifies the parameters used for PUE generation, FL training, the MD attack, and the baseline models used for comparison.
    \item \textbf{Chapter 5: Results} presents the empirical findings from the simulations. This chapter provides a quantitative analysis of the experiments, directly addressing the research questions by presenting data on the FL model's utility, the effectiveness of PUEs against the MD attack, and the trade-offs observed under various conditions.
    \item \textbf{Chapter 6: Discussion} provides an interpretation and analysis of the results. It discusses the implications of the findings, compares them with existing work in the field, and acknowledges the limitations of this study.
    \item \textbf{Chapter 7: Conclusion and Future Work} concludes the thesis. It summarizes the key contributions, offers final remarks on the research questions, and suggests potential directions for future research based on the outcomes of this work.
\end{itemize}


Of course. Here is a comprehensive markdown export focused specifically on the practical aspects of your project—the experimental framework, code, parameters, and key technical findings—to help you run experiments with a new model.

-----

# **Project Handover: Experimental Framework & Code**

## **1. Experimental Goal**

The primary goal of the experiments is to quantify the effectiveness of **Provably Unlearnable Examples (PUEs)** as a defense against **Model Distillation (MD) attacks** in a **Federated Learning (FL)** setting. This involves measuring the trade-off between the defense's success (how much it degrades the attacker's model) and the utility of the legitimate FL model.

-----

## **2. Core Experimental Workflow**

To get a complete set of results for one configuration (e.g., for a specific `epsilon` value), you must follow these four main steps:

1.  **Generate Poisoned Datasets:** For each of the 10 clients, run the `make_pue.py` script. This takes the client's clean data partition and generates a corresponding partition with PUE noise. This step needs to be done for each poison configuration (e.g., `min-min` vs. `min-max`, different `epsilon` values).
2.  **Train the Poisoned FL Model:** Use your FL simulation framework (Flower) to train a global model from scratch using the 10 poisoned client datasets generated in the previous step. The final saved global model is your "poisoned teacher model."
3.  **Run the Recovery Attack/Defense:** Take the final poisoned global model from Step 2 and run the modified `recover.py` script on it. This will produce your final "defended model."
4.  **Evaluate and Compare:**
      * Measure the **Clean Accuracy** of the defended model (from Step 3) on the clean test set.
      * Perform a **Model Distillation attack** on the defended model to train a student model. The final accuracy of this student model is the **Attack Success Rate (ASR)**.
      * Compare these two numbers against baselines (e.g., a model trained on clean data).

-----

## **3. Key Scripts & Required Modifications**

You are working with two main scripts from the `certified-data-learnability` repository. Both require minor but critical modifications to align with the paper's methodology.

### **`make_pue.py` (Poison Generation)**

  * **Purpose:** Creates the unlearnable noise for a given dataset.
  * **Key Flags for Active Attack:** To generate the stronger, active poison, you must use:
      * `--attack_type min-max`
  * **CRITICAL BUG:** The script has a `TypeError` when using `min-max`. To fix it, you must find the `train()` function call inside the `min-max` logic (line 511) and remove the extra `data_loader` argument.
      * **Change this:** `train(0, model, optimizer, scheduler, trainer, evaluator, ENV, data_loader)`
      * **To this:** `train(0, model, optimizer, scheduler, trainer, evaluator, ENV)`

### **`recover.py` (Defense/Recovery)**

  * **Purpose:** Loads a poisoned model and attempts to "recover" a clean version by fine-tuning it on a small set of clean data.

  * **CRITICAL MODIFICATION:** To replicate the "recovery from scratch" experiment shown in the paper's charts, you must modify the `main()` function. The goal is to load the checkpoint into a `poison_model` (the target) while leaving the `model` (the one to be trained) randomly initialized.

    ```python
    # === START OF MODIFICATION in recover.py's main() function ===

    # 'model' is already randomly initialized at the top of main()
    # Create a separate "poison_model" to be the target
    poison_model = config.model().to(device)

    if args.load_model:
        # Load the checkpoint ONLY into the poison_model
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=poison_model,  # <-- Change this from 'model'
                                     optimizer=None,
                                     alpha_optimizer=None,
                                     scheduler=None)
        logger.info("File %s loaded into TARGET model!" % (checkpoint_path_file))
    else:
        # This case is not useful for the paper's experiment
        logger.warning("Warning: --load_model not set. Target model will be random.")
        poison_model = copy.deepcopy(model)

    # The rest of the script then correctly uses the random 'model' for training
    # and the loaded 'poison_model' as the reference.

    # === END OF MODIFICATION ===
    ```

-----

## **4. Recommended Experimental Parameters**

This configuration is for a robust experiment designed to test a strong, active attack.

  * **Dataset:** CIFAR-10
  * **Model Architecture:** `ResNet-18` (or your new model)
  * **FL Setup:** 10 clients, each with a 5,000-image partition of the training data.

### **For `make_pue.py` (Poison Generation)**

  * `--attack_type`: `min-max` (This is the strong, active attack)
  * `--epsilon`: Start with `16` (which the script converts to 16/255). This is a strong budget. Consider `32` for a stress test.
  * `--num_steps`: `10`
  * `--step_size`: `4.0` (for `epsilon=16`) or `2.0` (for `epsilon=8`).
  * `--train_step`: `50` (to ensure the surrogate overfits well to the poison)

### **For `recover.py` (Defense/Recovery)**

  * `--load_model`: **Must be used.**
  * `--project`: **Must be used.**
  * `--recover_rate`: `0.2` (This will use 20% of the data for recovery).
  * `--use_train_subset`: **Use this flag.** This correctly tells the script to take 20% from the client's *training* partition (20% of 5,000 = **1,000 samples**), which aligns with the paper's description.