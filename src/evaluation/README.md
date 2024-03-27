# Evaluation Guideline

### Project summary

This project is for simulating the text-based adventure game, Jim Henson's Labyrinth: The Adventure game, based on the LLM-powered game manager. (a.k.a Goblin King in the game)

Especially, the key of this project is the function calling, which calls a user-defined function when it is needed considering the chat messages and the game states, to perform more fine-grained controls or directly use/update the game states. 

The goal of this project is how much the function calling can improve the performance of the game manager in terms of the management of the game states and game flow.

You can find the details of the project [here](https://docs.google.com/presentation/d/1cKqnnXSyjdAapoRutGWBpGQxWYKchQKYRBQcqCg4Wv4/edit?usp=sharing) (A little bit outdated, but it will still help you to understand the overall idea of the project.)

<br/>

---

### Gameplay data introduction

For evaluation, you are given JSON files and one file contains the gameplay for a single scene.

Although the evaluation script helps you to focus on the target responses/functions and required components you should refer to, we are attaching a few introductions of the gameplay data for further understanding.

```json
[
    {
        "scene": {},
        "players": [],
        "current_queries": [],
        "past_history": [],
        "generated": {},
        "function_calls": [
            {},
            {},
            ...
        ]
    },
       
    
    {
    	...        
    },
    
        
    ...
    
    {
        ...
    },
        
        
    {
        "game_result": "suceess/failure",
        "condition": "The detailed explanation of the game result."
    }
]
```

<br/>

The JSON file contains an array of multiple JSON objects.

Each JSON object represents one model completion given the scene state, player states, the past chat history, the current queries.

- `scene`: The current state of the scene. If includes the essential components of the scene, such as NPCs, environmental objects, random tables, and success/failure conditions, etc. (details on 1)
- `players`: The current states of the players. Each player character has its own state, such as persona, traits, flaws, and inventory, etc. (details on 2)
- `current_queries`: The current queries to process. Each new query starts when any player types a new message. After that, all messages until the game manager finishes processing all requests are considered as `current_queries`. (details on 3)
- `past_history`: The past chat history. Unlike `current_queries`, this only contains natural language messages without any NULL content or function execution results. (details on 4.)
- `generated`: The generated response from the AI game manager. This can be simply a natural language response, or function call request. (details on 5.)
- `function_calls`: The results of the called functions. This might contain multiple function results. (details on 6.)

<br/>

#### 1. Scene state

```
"scene": {
    "chapter": "Chapter name",
    "scene": "Scene name",
    "scene_summary": [..., ..., ...],
    "npcs": {..., ..., ...},
    "success_condition": "The success condition of the current scene.",
    "failure_condition": "The failure condition of the current scene.",
    "game_flow": [..., ..., ...],
    "environment": {..., ..., ...},
    "random_tables": {..., ..., ...},
    "consequences": "The consequence after finishing the scene.",
    "is_action_scene": true/false
}
```



<br/>

---

### Evaluation metrics



<br/>

---

### Additional notes



<br/>

---

**If you have any issues or questions, feel free to contact me: jwsong05@seas.upenn.edu.**

Thank you for your all hard works!