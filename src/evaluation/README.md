# Evaluation Guideline

### Project summary

This project is for simulating the text-based adventure game, Jim Henson's Labyrinth: The Adventure game, based on the LLM-powered game manager. (a.k.a Goblin King in the game) Especially, the key of this project is the function calling, which calls a user-defined function when it is needed considering the chat messages and the game states, to perform more fine-grained controls or directly use/update the game states. 

The goal of this project is how much the function calling can improve the performance of the game manager in terms of the management of the game states and game flow.

You can find the details of the project [here](https://docs.google.com/presentation/d/1cKqnnXSyjdAapoRutGWBpGQxWYKchQKYRBQcqCg4Wv4/edit?usp=sharing) (A little bit outdated, but it will still help you to understand the overall idea of the project.)

<br/>

---

### Gameplay data introduction

For evaluation, you are given JSON files and one file contains the gameplay for a single scene. Although the evaluation script helps you to focus on the target responses/functions and required components you should refer to, we are attaching a few introductions of the gameplay data for further understanding.

```json
[
    {
        "scene": {   },
        "players": [   ],
        "current_queries": [   ],
        "past_history": [   ],
        "generated": {   },
        "function_calls": [   ]
    },
       
    
    {
        
    },
    
    ...
    
    {
        
    },
        
        
    {
        "game_result": "suceess/failure",
        "condition": "The detailed explanation of the game result."
    }
]
```

<br/>

The JSON file contains an array of multiple JSON objects. Each JSON object represents one model completion given the scene state, player states, the past chat history, and the current queries.

- `scene`: The current state of the scene. If includes the essential components of the scene, such as NPCs, environmental objects, random tables, success/failure conditions, etc. (details on 1)
- `players`: The current state of the players. Each player character has its own state, such as persona, traits, flaws, inventory, etc. (details on 2)
- `current_queries`: The current queries to process. Each new query starts when any player types a new message. After that, all messages until the game manager finishes processing all requests are considered as `current_queries`. (details on 3)
- `past_history`: The past chat history. Unlike `current_queries`, this only contains natural language messages without any NULL content or function execution results. (details on 4.)
- `generated`: The generated response from the AI game manager. This can be simply a natural language response or function call request. (details on 5.)
- `function_calls`: The results of the called functions. This might contain multiple function results. (details on 6.)

Note that the last JSON object is just an indication of the game result, which is different from others.

- `game_result`: The result of the game. It can be either "success" or "failure".
- `condition`: The detailed explanation of the result. This can be the success/failure condition in the scene, or timeout.

<br/>

#### 1. Scene state

```json
{
    "chapter": "Chapter name",
    "scene": "Scene name",
    "scene_summary": [
    	"Summary sentence 1", 
        ...
        "Summary sentence n"
    ],
    "npcs": {
        "NPC name": {
            "kin": "Kin of the NPC",
            "persona": [
                "Persona sentence 1",
                ...
                "Persona sentence n"
            ],
            "goal": "Goal of the NPC",
            "trait": "Trait of the NPC",
            "flaw": "Flaw of the NPC"
        },
        "NPC name": {...},
        ...
    },
    "success_condition": "The success condition of the current scene.",
    "failure_condition": "The failure condition of the current scene.",
    "game_flow": [
    	"Game flow sentence 1",
        ...
        "Game flow sentence n"
    ],
    "environment": {
        "Object name": "Object description",
        ...
    },
    "random_tables": {
        "Table name": [
            "Entry 1",
            ...
            "Entry n"
        ],
        ...
    },
    "consequences": "The consequence after finishing the scene.",
    "is_action_scene": true/false
}
```

- `chapter`: The chapter name.
- `scene`: The scene name.
- `scene_summary`: This is an overall summary of the current scene. This is an array of strings, where one string is one summary sentence.
- `npcs`: This is an object for initialized NPCs. Each key is an NPC's name and the value is another JSON object which defines the specifications of that NPC.
  - `kin`: The kin of the NPC.
  - `persona`: The persona of the NPC. This is an array of strings, where one string is one persona sentence.
  - `goal`: The goal of the NPC.
  - `trait`: The trait of the NPC which would be helpful if it joins the player's party.
  - `flaw`: The flaw of the NPC which would be harmful if it joined the player's party. (Might leave the players due to this flaw.)
- `success_condition`: The condition for the players to win this scene.
- `failure_condition`: The condition for the players to lose this scene.
- `game_flow`: The intended game flow of the current game scene. This is an array of strings, in which one string is one progress sentence.
- `environment`: The environmental objects. This is a JSON object where each key is an object name and the value is the description of that object.
- `random_tables`: The random tables. This is a JSON object where each key is a table name and the value is an array of table entries.
- `consequences`: The consequence after finishing the scene.
- `is_action_scene`: The boolean value that indicates whether the action scene is currently activated or not.

<br/>

#### 2. Player state

```json
{
    "name": "Player name",
    "kin": "Kin of the player character",
    "persona": [
        "Persona sentence 1",
        ...
        "Persona sentence n"
    ],
    "goal": "Goal of the player character",
    "traits": {
        "Trait name": "Trait description",
        ...
    },
    "flaws": {
        "Flaw name": "Flaw description",
        ...
    },
    "inventory": {
        "Item name": "Item description",
        ...
    },
    "additional_notes": [
        "Note sentence 1",
        ...
        "Note sentence n"
    ]
}
```

- `name`: The name of the PC.
- `kin`: The kin of the PC.
- `persona`: The persona of the PC. This is an array of strings, where one string is one persona sentence.
- `goal`: The goal of the PC.
- `traits`: The traits of the PC. This is a JSON object where each key is a trait name and the value is the description of that trait.
- `flaws`: The flaws of the PC. This is a JSON object where each key is a flaw name and the value is the description of that flaw.
- `inventory`: The inventory of the PC. This is a JSON object where each key is an item name and the value is the description of that item.
- `additional_notes`: The additional notes for the PC. This is an array of strings, where one string is one note sentence. Each note might indicate specific behavior that the game manager should take when the PC does something. (e.g. Adding a flaw when a firey detaches its body part.)

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