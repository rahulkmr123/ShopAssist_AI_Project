import openai
import pandas as pd
import json

# This function initiates create the system and role conversation with Open AI model
def initialize_conversation():
    system_prompt = """
                You are ShopAssist, a friendly and smart laptop expert. Your goal is to help the user choose a laptop by asking the right questions and understanding their needs.
                
                🎯 Final Objective:
                At the end of the conversation, generate a natural sentence summarising the user's requirements in this format:
                "I need a laptop with <GPU intensity>, <Display quality>, <Portability>, <Multitasking>, <Processing speed> and a budget of <budget>."
                
                Each attribute should be either 'low', 'medium', or 'high' — depending on how important it is to the user. The budget should be extracted as a numeric value. Only include values you're confident about.
                
                📏 Rules:
                - Budget must be **at least 25000 INR**. If it's less, inform the user no options exist.
                - Never guess. Only assign a value if it's clearly implied.
                - If you're unsure about one or more attributes, ask specific follow-up questions to clarify.
                - Do not output the final summary sentence until you're confident about **all six** values.
                
                🧠 Chain of Thought:
                1. **Start by asking what they’ll use the laptop for.**
                   - E.g. gaming, editing, studying, browsing, etc.
                   - Based on this, you can infer some values (e.g. gaming → high GPU).
                2. **Ask clarifying questions** for missing attributes.
                   - E.g. Do they carry their laptop often? Do they multitask?
                   - Ask for their **budget** explicitly if not already given.
                3. **Once you’re confident about all values**, respond with the final summary sentence.
                
                💬 Sample conversation:
                User: Hi, I’m an editor.
                Assistant: Great! Editors usually need strong multitasking and high display quality. Do you work more on videos or photos?
                User: I mostly use After Effects.
                Assistant: Got it. That also needs high GPU and a strong processor. Do you carry your laptop while traveling?
                User: Not really, I work from one place.
                Assistant: Thanks! What’s your budget?
                User: 1.5 lakh INR.
                Assistant: Thank you. Here's a summary of your needs:
                I need a laptop with high GPU intensity, high Display quality, low Portability, high Multitasking, high Processing speed and a budget of 150000.
                
                Start with a short welcome message and ask the user what they plan to use the laptop for.
                """
    return [{"role": "system", "content": system_prompt.strip()}]


# The function encapsulates the chat Completion API call to Open AI
def get_chat_model_completions(messages):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",     # Most cost-effective model for this use case
            messages=messages,
            temperature=0.3,            # Lower for more consistent, focused responses
            max_tokens=150,             # Sufficient for laptop recommendations
            presence_penalty=0.1,       # Slight penalty to maintain context
            frequency_penalty=0.2,      # Reduce repetitive suggestions
            top_p=0.95                  # Slightly reduce randomness while maintaining quality
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in processing request: {str(e)}"


# The following fucntion checks the user input for content inputed for moderation check
def moderation_check(user_input):
    try:
        return "Flagged" if openai.moderations.create(input=user_input).results[0].flagged else "Not Flagged"
    except Exception as e:
        return "Error"

# The intent confirmation layer evaluates the output of the chat completion from Open AI API
def intent_confirmation_layer(response_assistant):
    prompt = """
    Verify if the input contains valid values for:
    GPU intensity, Display Quality, Portability, Multi tasking, Processing speed (must be low/medium/high)
    Budget (must be a number)
    Return Yes if all values are present and valid, No otherwise.
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Input: {response_assistant}"}
    ]

    try:
        confirmation = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1,  # More deterministic
            max_tokens=5      # Only need Yes/No
        )
        return confirmation.choices[0].message.content
    except Exception as e:
        return "Error"


# The intent confirmation layer evaluates the output of the chat completion from Open AI API
def get_user_requirement_string(response_assistant):
    prompt = """
            You are a helpful assistant that extracts structured user intent from a detailed message.
            
            The message contains a user’s laptop requirements explained in natural language. Your task is to extract and reformat the user's final needs into a structured single-line sentence with the following format:
            
            "I need a laptop with <GPU intensity> GPU intensity, <Display quality> display quality, <Portability> portability, <Multitasking> multitasking, <Processing speed> processing speed and a budget of <budget>."
            
            Guidelines:
            - Only use the words 'low', 'medium', or 'high' for GPU intensity, display quality, portability, multitasking, and processing speed.
            - Budget should be a number (e.g., 150000), without any currency symbol.
            - Do not add any commentary—just return the structured sentence.
            
            Example:
            
            Input:
            "Great! Based on your requirements, I have a clear picture of your needs. You prioritize low GPU intensity, high display quality, low portability, high multitasking, high processing speed, and have a budget of 200000 INR. Thank you for providing all the necessary information."
            
            Output:
            "I need a laptop with low GPU intensity, high display quality, low portability, high multitasking, high processing speed and a budget of 200000."
            """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Input: {response_assistant}"}
    ]
    confirmation = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return confirmation.choices[0].message.content.strip()

# Create custom function for using Open AI function calling
shopassist_custom_functions = [
    {
        'name': 'extract_user_info',
        'description': 'Extract laptop requirements from user input',
        'parameters': {
            'type': 'object',
            'required': ['GPU intensity', 'Display quality', 'Portability', 'Multitasking', 'Processing speed', 'Budget'],
            'properties': {
                'GPU intensity': {
                    'type': 'string',
                    'enum': ['low', 'medium', 'high'],
                    'description': 'Required GPU performance level'
                },
                'Display quality': {
                    'type': 'string',
                    'enum': ['low', 'medium', 'high'],
                    'description': 'Required display quality level'
                },
                'Portability': {
                    'type': 'string',
                    'enum': ['low', 'medium', 'high'],
                    'description': 'Required portability level'
                },
                'Multitasking': {
                    'type': 'string',
                    'enum': ['low', 'medium', 'high'],
                    'description': 'Required multitasking capability'
                },
                'Processing speed': {
                    'type': 'string',
                    'enum': ['low', 'medium', 'high'],
                    'description': 'Required processing speed level'
                },
                'Budget': {
                    'type': 'integer',
                    'minimum': 25000,
                    'description': 'Maximum budget in INR'
                }
            }
        }
    }
]

# Calls OpenAI API to return the function calling parameters
def get_chat_completions_func_calling(input, include_budget):
    try:
        messages = [
            {"role": "system", "content": "Extract laptop requirements from user input accurately"},
            {"role": "user", "content": input}
        ]

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=shopassist_custom_functions,
            function_call={'name': 'extract_user_info'},  # Force specific function
            temperature=0.1  # More consistent extraction
        )

        params = json.loads(response.choices[0].message.function_call.arguments)
        budget = params['Budget'] if include_budget else 0

        return extract_user_info(
            params['GPU intensity'],
            params['Display quality'],
            params['Portability'],
            params['Multitasking'],
            params['Processing speed'],
            budget
        )
    except Exception as e:
        return {"error": f"Function calling failed: {str(e)}"}

# The local function that we have written to extract the laptop information for user
def extract_user_info(GPU_intensity, Display_quality, Portability, Multitasking, Processing_speed, Budget):
    """

    Parameters:
    GPU_intensity (str): GPU intensity required by the user.
    Display_quality (str): Display quality required by the user.
    Portability (str): Portability required by the user.
    Multitasking (str): Multitasking capability required by the user.
    Processing_speed (str): Processing speed required by the user.
    Budget (int): Budget of the user.

    Returns:
    dict: A dictionary containing the extracted information.
    """
    return {
        "GPU intensity": GPU_intensity,
        "Display quality": Display_quality,
        "Portability": Portability,
        "Multitasking": Multitasking,
        "Processing speed": Processing_speed,
        "Budget": Budget
    }

# Compare and find laptops that match user requirements
def compare_laptops_with_user(user_requirements: dict) -> str:
    """Compare user requirements with available laptops and return top matches.

    Args:
        user_requirements (dict): User preferences containing GPU intensity, Display quality,
                                Portability, Multitasking, Processing speed, and Budget

    Returns:
        str: JSON string containing top 3 matching laptops with scores

    Raises:
        Exception: If there's an error in data processing or comparison
    """
    try:
        # Load and preprocess laptop data
        laptop_df = pd.read_csv('laptop_inventory.csv')
        laptop_df['laptop_feature'] = laptop_df['Description'].apply(product_map_layer)

        # Process budget and filter laptops
        budget = int(user_requirements.get('Budget', '0'))
        filtered_laptops = laptop_df.copy()
        filtered_laptops['Price'] = filtered_laptops['Price'].str.replace(',', '').astype(int)
        filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= budget].copy()

        # Define feature level mappings
        mappings = {'low': 0, 'medium': 1, 'high': 2}
        filtered_laptops['Score'] = 0

        # Calculate scores for each laptop
        for index, row in filtered_laptops.iterrows():
            laptop_values = get_chat_completions_func_calling(row['laptop_feature'], False)
            score = sum(
                1 for key, user_value in user_requirements.items()
                if key.lower() != 'budget'
                and mappings.get(laptop_values.get(key, '').lower(), -1) >=
                    mappings.get(user_value.lower(), -1)
            )
            filtered_laptops.loc[index, 'Score'] = score

        # Get top 3 matches
        top_laptops = (filtered_laptops
                      .drop('laptop_feature', axis=1)
                      .sort_values('Score', ascending=False)
                      .head(3))

        return top_laptops.to_json(orient='records')

    except Exception as e:
        return json.dumps({"error": f"Comparison failed: {str(e)}"})

def recommendation_validation(laptop_recommendation):
    data = json.loads(laptop_recommendation)
    data1 = []
    for i in range(len(data)):
        if data[i]['Score'] > 2:
            data1.append(data[i])

    return data1

def initialize_conv_reco(products):
    """Initialize conversation for laptop recommendations with product catalogue.

   Args:
       products (list): List of available laptop products with specifications

   Returns:
       list: Initial conversation context with system message

   Raises:
       ValueError: If products list is empty or invalid
   """
    system_message = f"""
    You are an intelligent laptop gadget expert and you are tasked with the objective to \
    solve the user queries about any product from the catalogue: {products}.\
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each laptop in the following format, in decreasing order of price of laptops:
    1. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>
    2. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>

    """
    conversation = [{"role": "system", "content": system_message }]
    return conversation

def product_map_layer(laptop_description: str) -> str:
    """Map laptop description to standardized feature classifications.

    Args:
        laptop_description (str): Raw laptop description text

    Returns:
        str: Standardized laptop feature string with classifications

    Raises:
        Exception: If classification process fails
    """
    try:
        delimiter = "#####"
        lap_spec = ("Laptop with (Type of the Graphics Processor) GPU intensity, "
                   "(Display Type, Screen Resolution, Display Size) display quality, "
                   "(Laptop Weight) portability, (RAM Size) multi tasking, "
                   "(CPU Type, Core, Clock Speed) processing speed")

        values = {'low', 'medium', 'high'}

        classification_rules = f"""
        GPU Intensity:
        - low: entry-level (integrated graphics, Intel UHD)
        - medium: mid-range (M1, AMD Radeon, Intel Iris)
        - high: high-end (Nvidia RTX)

        Display Quality:
        - low: below Full HD (1366x768)
        - medium: Full HD (1920x1080) or higher
        - high: 4K, Retina, HDR support

        Portability:
        - high: < 1.51 kg
        - medium: 1.51 - 2.51 kg
        - low: > 2.51 kg

        Multitasking:
        - low: 8GB, 12GB RAM
        - medium: 16GB RAM
        - high: 32GB, 64GB RAM

        Processing Speed:
        - low: Intel Core i3, AMD Ryzen 3
        - medium: Intel Core i5, AMD Ryzen 5
        - high: Intel Core i7+, AMD Ryzen 7+
        """

        example = {
            "input": "Dell Inspiron with Intel Core i5 @ 2.4GHz, 8GB RAM, "
                    "1920x1080 display, 2.5kg weight, Intel UHD GPU",
            "output": "Laptop with medium GPU intensity, medium display quality, "
                     "medium portability, low multitasking, medium processing speed"
        }

        prompt = f"""
        Classify laptop features based on description.
        Rules: {delimiter}\n{classification_rules}\n{delimiter}
        Example: {delimiter}\n{example['input']}\n{example['output']}\n{delimiter}
        Output format: {lap_spec}
        Note: Use only low/medium/high values.
        """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Classify: {laptop_description}"}
        ]

        return get_chat_model_completions(messages)

    except Exception as e:
        return f"Classification failed: {str(e)}"
