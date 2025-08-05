from flask import Flask, redirect, url_for, render_template, request
from HelperFunctions import (
    initialize_conversation,
    initialize_conv_reco,
    get_chat_model_completions,
    moderation_check,
    intent_confirmation_layer,
    compare_laptops_with_user,
    recommendation_validation,
    get_user_requirement_string,
    get_chat_completions_func_calling
)
import openai

openai.api_key=open("OpenAI_API_Key.txt",'r').read().strip()

app = Flask(__name__)

chat_conversation_history = []
conversation = initialize_conversation()
introduction = get_chat_model_completions(conversation)
chat_conversation_history.append({'bot':introduction})
top_3_laptops = None


@app.route("/")
def default_func():
    global chat_conversation_history, conversation, top_3_laptops
    return render_template("bot_page.html", name_xyz = chat_conversation_history)

@app.route("/end_conversation", methods = ['POST','GET'])
def end_conv():
    global chat_conversation_history, conversation, top_3_laptops
    chat_conversation_history = []
    conversation = initialize_conversation()
    introduction = get_chat_model_completions(conversation)
    chat_conversation_history.append({'bot':introduction})
    top_3_laptops = None
    return redirect(url_for('default_func'))

@app.route("/conversation", methods=['POST'])
def invite():
    """Handle conversation flow for laptop recommendations.

    Returns:
        redirect: Redirects to default route or end conversation
    """
    try:
        global chat_conversation_history, conversation, top_3_laptops, conversation_reco

        # Get and validate user input
        user_input = request.form["user_input_message"]
        prompt = ('Remember your system message and that you are an intelligent '
                 'laptop assistant. So, you only help with questions around laptop.')

        if moderation_check(user_input) == 'Flagged':
            chat_conversation_history.append({
                'bot': "Your message was flagged for violating our content policy. The conversation has been reset for your safety."
            })
            return redirect(url_for('end_conv'))

        # Initial conversation flow
        if top_3_laptops is None:
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input + prompt})
            chat_conversation_history.append({'user': user_input})

            # Get and validate assistant response
            response_assistant = get_chat_model_completions(conversation)
            if moderation_check(response_assistant) == 'Flagged':
                chat_conversation_history.append({
                    'bot': "Your message was flagged for violating our content policy. The conversation has been reset for your safety."
                })
                return redirect(url_for('end_conv'))

            # Check intent confirmation
            confirmation = intent_confirmation_layer(response_assistant)
            if moderation_check(confirmation) == 'Flagged':
                chat_conversation_history.append({
                    'bot': "Your message was flagged for violating our content policy. The conversation has been reset for your safety."
                })
                return redirect(url_for('end_conv'))

            if "No" in confirmation:
                # Continue conversation if requirements are incomplete
                conversation.append({"role": "assistant", "content": response_assistant})
                chat_conversation_history.append({'bot': response_assistant})
            else:
                # Process requirements and get recommendations
                response = get_user_requirement_string(response_assistant)
                result = get_chat_completions_func_calling(response, True)
                chat_conversation_history.append({
                    'bot': "Thank you for providing all the information. "
                          "Kindly wait, while I fetch the products: \n"
                })

                # Get and validate recommendations
                top_3_laptops = compare_laptops_with_user(result)
                validated_reco = recommendation_validation(top_3_laptops)

                if not validated_reco:
                    chat_conversation_history.append({
                        'bot': "Sorry, we do not have laptops that match your requirements. "
                              "Connecting you to a human expert. Please end this conversation."
                    })

                # Initialize recommendation conversation
                conversation_reco = initialize_conv_reco(validated_reco)
                recommendation = get_chat_model_completions(conversation_reco)

                if moderation_check(recommendation) == 'Flagged':
                    chat_conversation_history.append({
                        'bot': "Your message was flagged for violating our content policy. The conversation has been reset for your safety."
                    })
                    return redirect(url_for('end_conv'))

                # Update conversation history
                conversation_reco.append({
                    "role": "user",
                    "content": "This is my user profile" + response
                })
                conversation_reco.append({
                    "role": "assistant",
                    "content": recommendation
                })
                chat_conversation_history.append({'bot': recommendation})

        else:
            # Continue recommendation conversation
            conversation_reco.append({"role": "user", "content": user_input})
            chat_conversation_history.append({'user': user_input})

            response_asst_reco = get_chat_model_completions(conversation_reco)
            if moderation_check(response_asst_reco) == 'Flagged':
                return redirect(url_for('end_conv'))

            conversation.append({"role": "assistant", "content": response_asst_reco})
            chat_conversation_history.append({'bot': response_asst_reco})

        return redirect(url_for('default_func'))

    except Exception as e:
        print(f"Error in conversation handler: {str(e)}")
        return redirect(url_for('end_conv'))

if __name__ == '__main__':
    app.run(debug=True, host= "0.0.0.0", port=5001)
