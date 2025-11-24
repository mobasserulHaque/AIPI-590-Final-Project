# AIPI-590-Final-Project

**Reflect**: In-context alignment through model self-critique and revision.

![alt text](https://github.com/hbbell333/AIPI-590-Final-Project/blob/main/reflect_graphic.png "Logo Title Text 1")


During Reflect a single model critiques the ways in which it's own outputs are non-conformant. Based on this critique the model revises it's original output to correct the mistakes it identified.


**Constitution-conditioned base response**
The model first generates a constitution-conditioned base response. In this step, the entire constitution is passed to the model, along with a simple system prompt, and the user query. This base response is the starting point for the rest of the Reflect algorithm.  

**Self Evaluation**
Before running a full cycle of critique and revision, the model first evaluates how well the base response conforms to each principle in it's constitution. It scores the response on a 1-5 Likert scale for each principle in the constitution, using similar prompting to our multi-objective evaluation approach. If any principle scores below a user-defined threshold, these principles are flagged and the model will continue to the critique and revision step. 

**Critique and Revision**
During critique and revision the model is prompted to first generate a critique of it's base response and then to revise it based on the critique. The model's critique is only based on the principles that were flagged in step 2. The model can repeat steps 2-3 any number of times, stopping either when no principle falls below the threshold, or after a user-defined (though often one round is sufficient).  


![alt text](https://github.com/hbbell333/AIPI-590-Final-Project/blob/main/inference_graphic.png)


\* GPT5 codex was used to assist in creating streamlit web app and fix deployment issues (chat history shared in reference.txt attached)

\* Reflect algorithm code was adopted from work done in collaboration with Henry Bell, Mobasser Haque, Caroline Zhang, Samia Zaman, Dhaval Podtar, and Brandon Fain
