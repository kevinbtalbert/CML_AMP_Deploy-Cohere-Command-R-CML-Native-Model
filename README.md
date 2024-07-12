# CML_AMP_Deploy-Cohere-Command-R-CML-Native-Model
Deploy Cohere's Command R model using CML native models.

C4AI Command-R is a research release of a 35 billion parameter highly performant generative model. Command-R is a large language model with open weights optimized for a variety of use cases including reasoning, summarization, and question answering. Command-R has the capability for multilingual generation evaluated in 10 languages and highly performant RAG capabilities.

This project walks through a deployment and hosting of a Large Languge Model (LLM) within CML. The project is designed to be deployed as an AMP from the Cloudera AMP catalog.

### Model EULA Acceptance
Navigate to https://huggingface.co/CohereForAI/c4ai-command-r-v01 and accept the terms of the model. Then save the HF key by navigating to your profile and clicking "Settings", then "Access Tokens" and copy the key to your clipboard. Read only is sufficient access for this project.

### Site settings prerequisites
1. Go to Site Administration > Settings > Ephemeral Storage Limit (in GB) and set to 20GB minimum.
2. g4dn.4xlarge is the minimum recommended GPU type on AWS. It has 8 vCPUs and accounts for any overhead on top of 4 vCPUs that the model deployment needs.

### Deploy the model manually
Deploy the model by:
- Navigate to  Model Deployments
- Click `New Model`
- Give it a Name and Description
- Disable Authentication (for convenience)
- Select File `launch_model.py`
- Set Function Name `api_wrapper`
  - This is the function implemented in the python script which wraps text inference with the mistral7b model
- Set sample json payload
   ```
    {
    "prompt": "test prompt hello"
    }
   ```
- Pick Runtime
  - PBJ Workbench -- Python 3.9 -- Nvidia GPU -- 2023.08
- Set Resource Profile
  - At least 4CPU / 16MEM
  - 1 GPU
- Click `Deploy Model`
- Wait until it is Deployed

Test the Model.



Copyright (c) 2024 - Cloudera, Inc. All rights reserved.