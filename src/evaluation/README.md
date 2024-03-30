# Evaluation Guideline

### Project summary

This project is for simulating the text-based adventure game, Jim Henson's Labyrinth: The Adventure game, based on the LLM-powered game manager. (a.k.a Goblin King in the game) Especially, the key of this project is the function calling, which calls a user-defined function when it is needed considering the chat messages and the game states, to perform more fine-grained controls or directly use/update the game states. 

The goal of this project is how much the function calling can improve the performance of the game manager in terms of the management of the game states and game flow.

You can find the details of the project [here](https://docs.google.com/presentation/d/1cKqnnXSyjdAapoRutGWBpGQxWYKchQKYRBQcqCg4Wv4/edit?usp=sharing) (A little bit outdated, but it will still help you to understand the overall idea of the project.)

<br/>

---

### Survey instruction

For evaluation, you will be given survey links which enable you to give a score to each target response while checking the conversation between the players and the game manager and the game states on which the current scene is grounded. We attach a few instructions for the survey to help you understand what exactly each component represents and how you can follow the survey flow.

<br/>

First, one survey contains one gameplay data played by a party consisting of 4 player characters with different profiles. They interacted with the game manager, who is called "Goblin King" in the game, to clear the game scene. You should evaluate the generated responses, each of which is set to "target response", in terms of 3 different metrics. You will be given more specific instructions for each metric on the survey page.

For each evaluation, you will be given the context of the game, which includes **Starting game state**, **Starting player state** of each player, **Past chat history so far**, and **Current queries**. We introduce the details of these context components as follows:

<br/>

#### 1. Starting game state

The starting game state shows the initial game scene when the scene has started. Note that this is the starting state, so while proceeding with the game, the state might have been updated, which you should also consider for your evaluations. (The starting state does not contain the updates!)

- <u>**Chapter**</u>: The chapter name.
- <u>**Scene**</u>: The scene name.
- <u>**Scene Summary**</u>: Overall description of the current scene.
- <u>**NPCs**</u>: Initialized NPCs. The name of each NPC is tagged with **[Name]**. Below it, you can check the specifications of that NPC:
  - **Kin**: The kin of the NPC.
  - **Persona**: The persona of the NPC.
  - **Goal**: The goal of the NPC.
  - **Trait**: The trait of the NPC which would be helpful. (Trait name - Trait description)
  - **Flaw**: The flaw of the NPC which would be harmful. (Flaw name - Flaw description)
- <u>**Success Condition**</u>: The condition for the players to win this scene.
- <u>**Failure Condition**</u>: The condition for the players to lose this scene.
- <u>**Game Flow**</u>: The intended game flow of the current game scene. This might not have necessarily to be followed, but gives a useful hint to the game manager which is what should be done next during the game.
- <u>**Environment**</u>: The environmental objects. The name of each object is tagged with **[Name]**. Next to it, you can check the description of that object.
- <u>**Random Tables**</u>: The random tables. The name of each table is tagged with **[Name]**. Below it, you can check the actual entries of that table.
- <u>**Consequences**</u>: The consequences after finishing the scene.
- <u>**Action Scene**</u>: Indication of whether the action scene is currently activated or not. (True/False)

<br/>

#### 2. Starting player state

Starting player state shows the initialized state of each player character when the scene has started. Like the starting game state, these player states might have also been updated. So you should follow the dialogue and keep in mind these updates during the game for a correct evaluation. The name of each player is shown as **Starting player state of ...**

- <u>**Kin**</u>: The kin of the player.
- <u>**Persona**</u>: The persona of the player.
- <u>**Goal**</u>: The goal of the player.
- <u>**Traits**</u>: The traits of the player. The name of each trait is tagged with **[Name]**. Next to it, you can check the description of that trait.
- <u>**Flaws**</u>: The flaws of the player. The name of each flaw is tagged with **[Name]**. Next to it, you can check the description of that flaw.
- <u>**Inventory**</u>: The inventory of the player. The name of each item is tagged with **[Name]**. Next to it, you can check the description of that item.
- <u>**Additional Notes**</u>: The additional notes for the player. This contains any specific behavior that the game manager should take when the PC does something. (e.g. Adding a flaw when a firey detaches its body part.)

<br/>

#### 3. Current queries

The current queries contain the player messages and the game manager's responses or function call results. Although each player can speak only one message at a time, the game manager can generate multiple responses or call multiple functions sequentially until it thinks all necessary moves have been performed. For instance, if there are 4 players, the message list such as `P1 -> P2 -> P3 -> P4 -> GM(response) -> GM(function) -> GM(response)` is definitely possible. 

In this project, one "turn" indicates the whole interaction after the manager finishes its all responses. For example, `P1 -> P2 -> P3 -> P4 -> GM(response) -> GM(function) -> GM(response)` is one turn. `P1 -> P2 -> P3 -> P4 -> GM(response) -> P2 -> P4 -> P1 -> P3 -> GM(response) -> GM(function) -> GM(response)` includes two turns.

Note that whenever a function is called, there should always be a response from the game manager before calling the function. This message signals that a certain function should be called next. And if the response is for signaling a function, `function_call` flag indicates it. Additionally, if a function is called and finishes its job, it returns the result message. Then the game manager generates another message based on this result of the function. (The above example `P1 -> P2 -> P3 -> P4 -> GM(response) -> GM(function) -> GM(response)` is actually showing this case.)

```json
[
    {
        "role": "user",
        "name": "Name of the player",
        "content": "Message content"
    },
    {   },
    {   },
    ...
    {
        "role": "assistant",
        "name": "Goblin_King"/null,
        "content": "Message content"/null,
        "function_call": true/false
    },
    {
    	"role": "function",
     	"name": "Name of the function",
        "content": "Returned result message from the function"
    }
]
```

Each message is a JSON object:

- `role`: The role of the speaker. This can be `user`, `assistant`, or `function`.
- `name`: The name of the speaker. If `role` is `user`, this becomes the name of the player. If `role` is `function`, this becomes the name of function which has been called. If `role` is `assistant`, this becomes either `Goblin_King` or `null`.
  - If `function_call` is `true`, `name` is set to `null` by default.
- `content`: The content of the message. If `role` is `user`, this becomes the message from the player. If `role` is `function`, this becomes the returned result message after executing the function. If `role` is `assistant`, this becomes either `null` or a natural language response.
  - The content is not always `null ` even if `function_call` is set to `true`. Even if the message is for function call, it still can contain a normal response.
  - However, if the content is `null`, then it means `function_call` is `true` always.
- `function_call`: The boolean flag that indicates whether the message is for function call or a normal response.
  - If `function_call` is `true`, the next message from the game manager must be the function result.
  - It is impossible that the turn ends after the assistant message with `function_call` set to `true`. 

 <br/>

#### 4. Past history

The past history is also an array of JSON objects, where one object is one message. So the format is actually identical to the messages in `current_queries`.

The only difference here is `past_history` does not include any assistant message with `null` content and the function result. We assume that the message with `null` is only for calling the function, which is not needed after the function has been executed. Also, the function result is assumed to be injected into the next assistant response. So after a turn is finished, the system clears all these unnecessary messages and put the cleared turn into the past history list. As a result, `past_history` only contains the natural language interactions between the players and the game manager. 

<br/>

#### 5. Generated response

The generated response, which is represented as `generated`, is an assistant message which has been generated by the model given the current `scene`, `players`, `past_history` and `current_queries`. The format is the same as the one explained in **3. Current queries**. This response will go into `current_queries` to start a new turn by the players.

<br/>

#### 6. Function result

The function results are marked as `function_calls`. (which should originally be `function_results`... Sorry for the confusion.) This is an array of JSON objects, where one JSON object includes a result from one function. In other words, it is possible that multiple functions can be run concurrently. (For this project, you don't have to worry about it. You may assume that the system runs only one function at a time...)

```json
[
    {
       "result": {
           "role": "function",
           "name": "Name of the function",
           "content": "Returned result message from the function"
       },
       "arguments": {
           "Argument name": "Argument value",
           "Argument name": "Argument value",
           ...
       },
       "intermediate_results": {
           "Intermediate task name": "Result",
           "Intermediate task name": "Result",
           ...
       }
    },
    ...
]
```

- `result`: This is actually identical to the one described in **3. Current queries**. After finishing the function, only `result` is put into `current_queries` for the next generation.
- `arguments`: This is a JSON object which contains the arguments used for executing the function. Each key is the name of the argument and the value is the actual value of that argument.
  - Some functions might not have any arguments.
- `intermediate_results`: Some functions have a few sub-tasks which are needed to finish the main task. For example, `activate_test` determines whether a test is improved or not. `use_environment` determines whether the object can be retrieved or not. `use_random_table` actually randomly samples the table entries and removes the entries or the table if necessary. `intermediate_results` contain the results of these intermediate tasks for more detailed evaluations. Each key is the name of the sub-task and the value is the result of that task.
  - Some functions might not have any intermediate results.


<br/>

---

### Additional notes



<br/>

---

**If you have any issues or questions, feel free to contact me: jwsong05@seas.upenn.edu.**

Thank you for your all hard works!