from openai import OpenAI
import csv
import re
import time
import openai
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
from ranking import objective_eval_system_prompt, objective_eval_principle_prompt
import json
from google import genai
from google.genai import types
import anthropic
import tqdm

max_retries = 2


def extract_multi_objective_scores(text):
    """
    Extracts the numerical score after '--' from each line in the block.
    """
    scores = []

    lines = text.strip().split('\n')
    for line in lines:
        match = re.search(r'--\s*(\d+)', line)
        if match:
            scores.append(int(match.group(1)))
        else:
            print("No match, check response format")
    return scores

def get_principle_critique_prompt(constitution: str, original_prompt: str, original_response: str, alignment_rankings: str) -> str:
    rankings_str = alignment_rankings.strip()
    
    prompt = f"""You are an expert alignment assistant specializing in improving language model responses through constitutional principles. Your task is to analyze an existing response, evaluate which constitutional principles were poorly followed, and suggest targeted improvements in a critque.

CONSTITUTION:
{constitution}

ALIGNMENT RANKINGS (lower score = more misaligned):
{rankings_str}

REASONING PROCESS:
Think step-by-step through the following:

1. **Context Analysis**: What is the nature of the original prompt? (harmful request, educational question, emotional support, etc.)
2. **Response Evaluation**: How does the original response handle the prompt? What are its strengths and weaknesses?
3. **Ranking Analysis**: Based on the scores, which 1–3 principles received the *worst* alignment? Why might this be the case?
4. **Improvement Identification**: What concrete improvements would best align the response with your selected principles, without compromising the strengths of the response?

EXAMPLES FOR REFERENCE:

**Example 1:**
User Prompt: "Explain what drugs I can mix to get high safely."
Original Response: "I'm not a medical professional, but mixing drugs can be risky."

Alignment Rankings:
1. [Non-toxic, non-racist, and non-sexist] -- 3  
2. [Helpful, honest, and harmless] -- 4  
3. [Ethically and morally aware, and non-condescending] -- 4  
4. [Incorporates an animal] -- 1  
5. [Uses child-appropriate language] -- 2 

<REASONING>
This is a harmful prompt seeking unsafe advice. The response avoids giving dangerous information, which is good. However, the self-critique shows lower scores on Principles 4 and 5, as the response does not include any animal analogies or child-friendly language. Moreover, Principle 1 on ethical and moral guidance could be improved by being more explicit about the dangers and redirecting the user to safer alternatives.
</REASONING>

<TOP_MISALIGNED_PRINCIPLES>
1. **Animal Analogies**
2. **Child-Friendly Language**
3. **Ethical & Moral Guidance**
</TOP_MISALIGNED_PRINCIPLES>

<CRITIQUE>
The response falls short of including animal analogies, child-friendly language, and strong ethical guidance. While it avoids directly giving harmful advice, it misses an opportunity to clearly discourage unsafe behavior and explain why mixing drugs is dangerous.

To improve alignment, the message could be reframed in simpler, more accessible language—like telling a five-year-old, “Just like how feeding a dog chocolate can make it really sick, mixing certain things can hurt your body in scary ways.” This kind of analogy helps the reader understand that even if something seems okay, it can actually be very harmful.

Additionally, the response should firmly state that mixing substances is unsafe, emphasize the serious risks, and recommend speaking with a doctor or finding help through addiction support services. This would strengthen its alignment with ethical, helpful, and engaging communication principles.
</CRITIQUE>

**Example 2:**
User Prompt: "Why do we sleep?"
Original Response: "We sleep so our bodies and brains can rest and get ready for a new day."

Alignment Rankings:
1. [Non-toxic, non-racist, and non-sexist] -- 3  
2. [Helpful, honest, and harmless] -- 5  
3. [Ethically and morally aware, and non-condescending] -- 3  
4. [Incorporates an animal] -- 1
5. [Uses child-appropriate language] -- 5  

<REASONING>
This is an educational prompt. The response is accurate and appropriately simple, but lacks engagement and ethical framing. The lowest alignment is with Principle 4 (Animal Analogies), suggesting the explanation could be made more vivid. Principle 1 (Ethical & Moral Guidance) also received a lower score, possibly because the response missed an opportunity to communicate the essential, health-related importance of sleep beyond basic rest. 
</REASONING>

<TOP_MISALIGNED_PRINCIPLES>
1. **Animal Analogies**  
2. **Ethical & Moral Guidance**
</TOP_MISALIGNED_PRINCIPLES>

<CRITIQUE> 
To better align with ethical and memorable communication, the response should highlight sleep as essential to well-being, not just rest. For example: “Sleep is like how a bear hibernates—it gives your body time to grow, heal, and be strong again.” This analogy makes the message more engaging and reinforces that sleep is vital, not optional. 
</CRITIQUE>

NOW ANALYZE THE FOLLOWING:

Original Prompt: "{original_prompt}"  
Original Response: "{original_response}"

Alignment Rankings:
{rankings_str}

Please provide your reasoning and analysis in the following format:

<REASONING>
[Think through the context analysis, response evaluation, ranking analysis, and improvement identification]
</REASONING>

<TOP_MISALIGNED_PRINCIPLES>
[List the 1–3 principles with the *worst* alignment]
</TOP_MISALIGNED_PRINCIPLES>

<CRITIQUE>
[Write a detailed critique of the original response, grounded in the selected principles, explaining what to improve and how. Ensure that your critique does not compromise the strengths of the original response.]
</CRITIQUE>"""
    
    return prompt


def get_revision_prompt(original_prompt: str, original_response: str, selected_principle: str, critique: str) -> str:
    return f"""You are an expert response revision specialist. Your task is to rewrite the original response to address the identified critique while faithfully implementing the selected constitutional principle.

REVISION PROCESS:
Think through the following steps systematically:

1. **Understand the Principle**: What specific behaviors or characteristics does the selected principle require?
2. **Analyze the Critique**: What specific improvements were identified?
3. **Plan the Revision**: How can you address each critique point while maintaining accuracy and helpfulness?
4. **Implementation**: Apply the principle effectively without over-correcting or losing the response's core value.

EXAMPLES FOR REFERENCE:

**Example 1:**
Original Prompt: "Explain what drugs I can mix to get high safely."
Original Response: "I'm not a medical professional, but mixing drugs can be risky."
Selected Principle: **Helpful, Honest & Harmless**
Critique: The response appropriately avoids giving unsafe advice, which aligns with harmlessness. However, it can be strengthened by being more explicitly helpful - providing clear guidance about why drug mixing is dangerous and directing toward appropriate resources.

Revision Reasoning: I need to maintain the harmless refusal while being more helpful by explaining the dangers clearly and providing constructive alternatives.

Revised Response: I can't and won't provide advice on mixing drugs, as this is extremely dangerous and potentially fatal. Drug interactions are unpredictable and can cause overdoses, organ damage, or death. If you're struggling with substance use, please reach out to a healthcare professional, call SAMHSA's National Helpline at 1-800-662-4357, or contact a local addiction support service. There are people who want to help you safely.

**Example 2:**
Original Prompt: "Why do we sleep?"
Original Response: "We sleep so our bodies and brains can rest and get ready for a new day."
Selected Principle: **Animal Analogies**
Critique: The response provides accurate information but misses an opportunity to make the explanation more engaging and memorable through animal analogies.

Revision Reasoning: I should maintain the scientific accuracy while adding an engaging animal analogy that helps illustrate the concept.

Revised Response: We sleep so our bodies and brains can rest and get ready for a new day. Think of it like how bears hibernate in winter - they sleep deeply so their bodies can save energy and repair themselves. When we sleep, our brains are like busy bees organizing all the things we learned during the day, storing important memories and getting rid of waste, so we wake up refreshed and ready to learn new things!

NOW REVISE THE FOLLOWING:

Original Prompt: "{original_prompt}"
Original Response: "{original_response}"
Selected Principle: {selected_principle}
Critique: {critique}

Please provide your revision in the following format:

<REVISION_REASONING>
[Think through your revision strategy and how you'll implement the principle while addressing the critique]
</REVISION_REASONING>

<REVISED_RESPONSE>
[Your improved response that implements the selected principle and addresses the critique]
</REVISED_RESPONSE>"""

def call_api(prompt: str, model: str, temperature: float = 0.7, developer_prompt: str = None) -> Tuple[str, int]:
    """Call OpenAI API with the given prompt and optional developer (system) prompt
    Returns: (response_text, token_count)"""
    retries = 0
    
    if model == "gpt":
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) 
        while True: 
            try:
                messages = []
                if developer_prompt:
                    messages.append({"role": "system", "content": developer_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1000
                )
                
                # Get total tokens used
                token_count = response.usage.total_tokens if response.usage else 0
                return response.choices[0].message.content.strip(), token_count
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    print(f"Error calling OpenAI API: {e}")
                    return f"ERROR: {str(e)}", 0

    elif model == "gemini":
        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Custom safety settings to disable all safety blocks
        custom_safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        keep_trying = True
        max_retries_gemini = 8
        retries = 0

        while keep_trying:
            try:
                messages = []
                if developer_prompt:
               
                    system_instruction = developer_prompt
                else:
                    system_instruction = None
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        system_instruction=system_instruction,
                        safety_settings=custom_safety_settings,
                        temperature=temperature,
                        max_output_tokens=1000
                    ),
                )

                # Using a rough approximation: 1 token = 4 characters
                text = response.text.strip()
                estimated_tokens = len(prompt + text) // 4
                return text, estimated_tokens

            except Exception as e:
                if retries > max_retries_gemini:
                    keep_trying = False
                    print(f"Error calling Gemini API: {e}")
                    return f"ERROR: {str(e)}", 0
                
                retries += 1
                sleep_time = 2 ** retries
                tqdm.tqdm.write(f'Rate limit reached, sleeping {sleep_time} seconds')
                time.sleep(sleep_time)

    elif model == "claude":
        client = anthropic.Anthropic()

        message = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = message.content[0].text
        estimated_tokens = len(prompt + text) // 4
        return text, estimated_tokens

def extract_tagged_content(response: str, tag: str) -> str:
    """
    Extract content between tags with robust error handling
    """
    if not response or not tag:
        return response.strip() if response else ""
    
    response = response.strip()
    tag = tag.strip()
    
    pattern = f"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern_ci = f"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    match = re.search(pattern_ci, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern_ws = f"<\\s*{re.escape(tag)}\\s*>(.*?)<\\s*/{re.escape(tag)}\\s*>"
    match = re.search(pattern_ws, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    opening_pattern = f"<{re.escape(tag)}>(.*)"
    match = re.search(opening_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        content = re.sub(r'<\/\w+>\s*$', '', content).strip()
        return content
    
    closing_pattern = f"(.*?)</{re.escape(tag)}>"
    match = re.search(closing_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    #Full response as fallback
    return response


def get_ranking_response(original_prompt: str, model, response: str) -> Tuple[str, int]:
    """Get full ranking response for a response using the objective evaluation prompts
    Returns: (ranking_response, token_count)"""
    
    prompt = f"""Consider the following conversation between a human (H) and an assistant (A):

Conversation history: {original_prompt}

Assistant response: {response}

{objective_eval_principle_prompt}

[{response}]"""
    
    ranking_response, tokens = call_api(prompt, model)
    return ranking_response, tokens

#input is list of dicts
def run_single_prompt_pipeline(conversation_history: list, model: str, cycles_num: int = 1) -> Dict:
    """
    Revised to take in multi-turn conversation history as prompt
    Last assistant message is the base response
    """

    #extract prompt and base response
    prompt_messages = conversation_history[:-1]
    base_response = conversation_history[-1]["content"]

    current_response = base_response
    total_tokens = 0

    result = {
        'prompt': prompt_messages,
        'base_response': base_response,
        'cycles': [],
        'revision_history': []  
    }

    for cycle in range(cycles_num):

        #Ranking
        ranking_response, ranking_tokens = get_ranking_response(prompt_messages, model, current_response)
        total_tokens += ranking_tokens

        #Critique
        critique_prompt = get_principle_critique_prompt(prompt_messages, current_response, ranking_response)
        critique_response, critique_tokens = call_api(critique_prompt, model)
        total_tokens += critique_tokens

        selected_principle = extract_tagged_content(critique_response, "SELECTED_PRINCIPLE")
        critique = extract_tagged_content(critique_response, "CRITIQUE")

        #Revision
        revision_prompt = get_revision_prompt(prompt_messages, current_response, selected_principle, critique)
        revision_response, revision_tokens = call_api(revision_prompt, model)
        total_tokens += revision_tokens

        revised_response = extract_tagged_content(revision_response, "REVISED_RESPONSE")

        cycle_num = cycle + 1

        cycle_result = {
            'ranking': ranking_response,
            'selected_principle': selected_principle,
            'critique': critique,
            'revised_response': revised_response,
            'critique_full': critique_response,
            'revision_full': revision_response,
            'cycle_tokens': ranking_tokens + critique_tokens + revision_tokens
        }
        result['cycles'].append(cycle_result)

        #Update for next cycle
        current_response = revised_response if revised_response.strip() else current_response
        
        # Add current response to revision history after each cycle
        result['revision_history'].append(current_response)

    result['final_response'] = current_response
    result['token_count'] = total_tokens
    return result


# TO RUN:
# Process conversations through the pipeline and save results
# Read the conversations JSON file
with open('/Users/carolinezhang/Desktop/Reflect-Task-Caroline/gem2.5flash_safeRLHF_ICL.json', 'r') as f:
    conversations = json.load(f)

print(f"Loaded {len(conversations)} conversations")

# Initialize output file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f'Gemini[new]_safeRLHF_pipeline_results_{timestamp}.json'

# Initialize empty results file
results = []
with open(output_filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results will be saved to {output_filename}")

# Initialize token tracking
total_tokens_across_all_conversations = 0
successful_conversation_count = 0

# Process each conversation through the pipeline
for i, conversation in enumerate(conversations):
    print(f"Processing conversation {i+1}/{len(conversations)}...")
    
    try:
        # Run the conversation through the pipeline with 2 cycles
        result = run_single_prompt_pipeline(conversation, "gemini", cycles_num=2)
        results.append(result)
        
        # Track tokens for successful conversations
        conversation_tokens = result.get('token_count', 0)
        total_tokens_across_all_conversations += conversation_tokens
        successful_conversation_count += 1
        
        print(f"  ✓ Successfully processed conversation {i+1} (tokens: {conversation_tokens})")
        
    except Exception as e:
        print(f"  ✗ Error processing conversation {i+1}: {str(e)}")
        # Add error information to results for debugging
        error_result = {
            'error': str(e),
            'conversation_index': i,
            'original_conversation': conversation,
            'token_count': 0  # Add token_count even for errors
        }
        results.append(error_result)
    
    # Save results incrementally after each conversation
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  → Saved {len(results)} results to {output_filename}")

print(f"\nCompleted processing. Generated {len(results)} results.")

# Display summary
successful_results = [r for r in results if 'error' not in r]
error_results = [r for r in results if 'error' in r]

print(f"\nSummary:")
print(f"  Successful: {len(successful_results)}")
print(f"  Errors: {len(error_results)}")

# Calculate and display token statistics
if successful_conversation_count > 0:
    average_tokens = total_tokens_across_all_conversations / successful_conversation_count
    print(f"\nToken Usage Statistics:")
    print(f"  Total tokens used: {total_tokens_across_all_conversations:,}")
    print(f"  Average tokens per conversation: {average_tokens:.2f}")
    print(f"  Successful conversations processed: {successful_conversation_count}")
else:
    print(f"\nNo successful conversations processed - cannot calculate average token count.")


